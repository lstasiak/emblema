import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.training.device_generator import DeviceGenerator
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.torch_precision import TorchPrecision
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.adapters.training.training_checkpoint import TrainingCheckpoint
from emblema.pretraining.domain.exceptions import DivergedRunError, IncompatibleCheckpointError
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_position import RunPosition
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention

# Validation masks are drawn from a generator of their own per batch, so that the curve moves with
# the model rather than with the masks. The stride keeps those seeds clear of the run's own.
VALIDATION_SEED_STRIDE = 1_000


@dataclass
class _Session:
    """Everything one run holds while it runs; assembled before the first epoch, mutated by it."""

    configuration: ExperimentConfiguration
    corpus: TrainingCorpus
    signature: RunSignature
    precision: TorchPrecision
    model: MaskedReconstruction
    loss: ReconstructionLoss
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    training: WindowLoader
    validation: WindowLoader
    masking: TokenMasking
    draws: torch.Generator
    position: RunPosition


class TorchTrainingRuntime:
    """Trains the objective in this process, on whatever device the machine has.

    The loop is the ordinary one — a forward pass under autocast, gradients accumulated over as
    many micro-batches as the budget asks for, one optimiser step, one scheduler step — and what
    it adds is that it can be stopped and picked up. That is what the checkpoints are for: written
    on a step boundary, when no gradient is pending, and carrying the position so a resumed run
    skips exactly the micro-batches the interrupted one had consumed.

    Masks are drawn on the host and moved, as the batches are, so a run on any device hides the
    same tokens; the device is where the arithmetic happens, never where the randomness comes
    from.
    """

    def __init__(
        self, store: ArtifactStore, *, device: str | None = None, num_workers: int = 0
    ) -> None:
        """Train against ``store``, which keeps the checkpoints and the model.

        Args:
            store: Where checkpoints and the trained model are written.
            device: Where the arithmetic happens; the machine's accelerator unless given.
            num_workers: Processes collating batches; zero collates in this one.
        """
        self._store = store
        self._device = available_device() if device is None else device
        self._generator = DeviceGenerator(self._device)
        self._num_workers = num_workers

    @property
    def device(self) -> str:
        return self._device

    def train(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        """Train the configuration over the corpus, an outcome per epoch.

        The run is assembled here and trained by the iterator, so a configuration this machine
        cannot honour, or a checkpoint from another run, is refused before anything is trained.

        Raises:
            IncompatibleCheckpointError: If the checkpoint belongs to another run, or to one that
                has finished.
            UnsupportedPrecisionError: If this device cannot train at the declared precision.
            DivergedRunError: While the run trains, if a loss or a gradient stops being finite.
        """
        return self._epochs(self._assemble(configuration, corpus, resume_from))

    def restore(self, ref: ArtifactRef) -> MaskedReconstruction:
        """The model stored under ``ref``, rebuilt on the host in evaluation mode.

        What a run produced is read back through the store rather than kept in memory, so
        whatever is diagnosed afterwards is what was actually written.

        Raises:
            UnreadableTrainedModelError: If the artifact is not a trained model of ours.
        """
        return TrainedModel.read(self._store.get(ref)).build()

    def _assemble(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None,
    ) -> _Session:
        budget = configuration.budget
        precision = TorchPrecision(configuration.precision, self._device)
        signature = RunSignature.of(configuration, corpus)
        # Built on the host from the run's seed, so the same seed gives the same initial weights
        # whatever the device, and moved afterwards.
        torch.manual_seed(budget.seed)
        model = MaskedReconstruction(
            SetEncoder.for_vocabulary(
                configuration.architecture, corpus.vocabulary_size, dropout=configuration.dropout
            ),
            decoder_layers=configuration.decoder_layers,
            dropout=configuration.dropout,
        ).to(self._device)
        loss = ReconstructionLoss(configuration.loss)
        training = WindowLoader(
            corpus.training,
            batch_size=budget.batch_size,
            seed=budget.seed,
            num_workers=self._num_workers,
        )
        validation = WindowLoader(
            corpus.validation, batch_size=budget.batch_size, seed=budget.seed, shuffle=False
        )
        # No weight decay, said rather than inherited: a regularisation nobody chose should not be
        # one a reader has to look up. The day it varies it becomes a field of the experiment.
        optimiser = torch.optim.Adam(model.parameters(), lr=budget.learning_rate, weight_decay=0.0)
        session = _Session(
            configuration=configuration,
            corpus=corpus,
            signature=signature,
            precision=precision,
            model=model,
            loss=loss,
            optimiser=optimiser,
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimiser, configuration.schedule(len(training)).factor
            ),
            scaler=precision.scaler(),
            training=training,
            validation=validation,
            masking=TokenMasking(configuration.masking),
            draws=torch.Generator().manual_seed(budget.seed),
            position=RunPosition.start(),
        )
        if resume_from is not None:
            self._resume(session, resume_from)
        return session

    def _resume(self, session: _Session, ref: ArtifactRef) -> None:
        """Put a checkpoint's state back into an assembled run, and refuse one that does not fit."""
        checkpoint = TrainingCheckpoint.read(self._store.get(ref), signature=session.signature)
        position = checkpoint.position
        batches = len(session.training)
        if position.epoch >= session.configuration.budget.epochs or (
            position.epoch == session.configuration.budget.epochs - 1
            and position.batches >= batches
        ):
            raise IncompatibleCheckpointError(
                f"the run this checkpoint comes from has finished: epoch {position.epoch} of "
                f"{session.configuration.budget.epochs}, {position.batches} of {batches} batches"
            )
        session.model.load_state_dict(checkpoint.model)
        session.optimiser.load_state_dict(checkpoint.optimiser)
        session.scaler.load_state_dict(checkpoint.scaler)
        session.draws.set_state(checkpoint.masks)
        torch.set_rng_state(checkpoint.seeds)
        self._generator.restore(checkpoint.device_seeds)
        # The schedule is a function of the step, so the scheduler is put where the run left it
        # rather than stepped there: its constructor advances the count it is given by one.
        for group in session.optimiser.param_groups:
            group.setdefault("initial_lr", session.configuration.budget.learning_rate)
        session.scheduler = torch.optim.lr_scheduler.LambdaLR(
            session.optimiser,
            session.configuration.schedule(batches).factor,
            last_epoch=position.steps - 1,
        )
        session.position = position

    def _epochs(self, session: _Session) -> Iterator[EpochOutcome]:
        budget = session.configuration.budget
        batches = len(session.training)
        while session.position.epoch < budget.epochs:
            if session.position.batches >= batches:
                # The checkpoint was written on the last step of its epoch, so that epoch is
                # behind and the run that wrote it reported the outcome.
                session.position = session.position.next_epoch()
                continue
            epoch = session.position.epoch
            started = time.perf_counter()
            training, checkpoint = self._epoch(session, batches)
            validation, hidden_ratio = self._validate(session)
            final = epoch + 1 == budget.epochs
            yield EpochOutcome(
                epoch=epoch,
                training_loss=training,
                validation_loss=validation,
                hidden_ratio=hidden_ratio,
                seconds=time.perf_counter() - started,
                checkpoint=checkpoint,
                backbone=self._write_model(session) if final else None,
            )
            session.position = session.position.next_epoch()

    def _epoch(self, session: _Session, batches: int) -> tuple[float, ArtifactRef | None]:
        """One pass over the training windows from where the run stands.

        Returns the error over its hidden tokens and the last checkpoint written during it, which
        is what a run interrupted in the next epoch would be picked up from.
        """
        budget = session.configuration.budget
        skip = session.position.batches
        checkpoint: ArtifactRef | None = None
        session.model.train()
        epoch_error, epoch_tokens, group_tokens = 0.0, 0, 0
        for index, on_host in enumerate(session.training.batches_of(session.position.epoch)):
            if index < skip:
                continue
            masks = session.masking.draw(on_host, session.draws)
            batch, drawn = on_host.to(self._device), masks.to(self._device)
            with session.precision.autocast():
                error, scored = session.loss.summed(
                    session.model(batch, drawn), batch, drawn.hidden
                )
            summed = float(error.detach())
            if not math.isfinite(summed):
                raise DivergedRunError(
                    f"the loss of batch {index} in epoch {session.position.epoch} is {summed}, "
                    f"after {session.position.steps} steps"
                )
            # Accumulated micro-batches make one gradient, and the error is summed rather than
            # averaged per batch so that each hidden token weighs the same whichever batch it
            # landed in. The sum is divided out of the gradients once, below, before the step.
            session.scaler.scale(error).backward()
            hidden = int(scored)
            group_tokens += hidden
            epoch_error += summed
            epoch_tokens += hidden
            stepped = budget.takes_a_step_at(index, batches)
            if stepped:
                self._average_gradients(session, group_tokens)
                # A gradient scaler skips a step whose gradients overflowed, which is what it is
                # for; without one, such a step would write non-finite weights into the model and
                # into the next checkpoint, so it is refused here instead.
                if not session.precision.scales_gradients:
                    self._refuse_non_finite_gradients(session)
                session.scaler.step(session.optimiser)
                session.scaler.update()
                session.optimiser.zero_grad(set_to_none=True)
                session.scheduler.step()
                group_tokens = 0
            session.position = session.position.after_batch(stepped=stepped)
            if stepped and session.configuration.checkpoint.due_at(session.position.steps):
                checkpoint = self._write_checkpoint(session)
        return epoch_error / max(epoch_tokens, 1), checkpoint

    @staticmethod
    def _average_gradients(session: _Session, tokens: int) -> None:
        """Turn the summed gradients of a group of micro-batches into the mean over its tokens."""
        if tokens < 1:
            return
        with torch.no_grad():
            for parameter in session.model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= tokens

    @staticmethod
    def _refuse_non_finite_gradients(session: _Session) -> None:
        gradients = [p.grad for p in session.model.parameters() if p.grad is not None]
        if gradients and not bool(torch.stack([g.isfinite().all() for g in gradients]).all()):
            raise DivergedRunError(
                f"a gradient is not finite at step {session.position.steps + 1} of epoch "
                f"{session.position.epoch}, at {session.configuration.precision}"
            )

    def _validate(self, session: _Session) -> tuple[float, float]:
        """The loss on the validation windows and the share of their observed tokens it scored.

        The masks are drawn per batch from a generator the seed fixes, so every epoch is scored
        under the same hidden tokens: what moves between epochs is the model.
        """
        session.model.eval()
        total, hidden_tokens, observed = 0.0, 0, 0
        with torch.no_grad():
            for index, on_host in enumerate(session.validation.batches_of(0)):
                draws = torch.Generator().manual_seed(
                    session.configuration.budget.seed * VALIDATION_SEED_STRIDE + index
                )
                masks = session.masking.draw(on_host, draws)
                batch, drawn = on_host.to(self._device), masks.to(self._device)
                scored = int(drawn.hidden.sum())
                with session.precision.autocast():
                    predicted = session.model(batch, drawn)
                total += float(session.loss(predicted, batch, drawn)) * scored
                hidden_tokens += scored
                observed += int((~batch.padding_mask).sum())
        return total / max(hidden_tokens, 1), hidden_tokens / max(observed, 1)

    def _write_checkpoint(self, session: _Session) -> ArtifactRef:
        """The whole state of the run, kept only as long as a dropped session might want it."""
        checkpoint = TrainingCheckpoint(
            signature=session.signature,
            position=session.position,
            model=session.model.state_dict(),
            optimiser=session.optimiser.state_dict(),
            scaler=session.scaler.state_dict(),
            masks=session.draws.get_state(),
            seeds=torch.get_rng_state(),
            device_seeds=self._generator.state(),
        )
        return self._store.put(checkpoint.to_bytes(), Retention.TRANSIENT)

    def _write_model(self, session: _Session) -> ArtifactRef:
        """The weights the run was for, kept until someone removes them."""
        model = TrainedModel.of(
            session.configuration, session.corpus.vocabulary_size, session.model
        )
        return self._store.put(model.to_bytes(), Retention.DURABLE)
