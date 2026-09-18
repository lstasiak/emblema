import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

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
from emblema.pretraining.domain.training.corpus_validation import CorpusValidation
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_position import RunPosition
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.adapters.loaders.interleaved_loader import InterleavedLoader
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention

# Validation masks are drawn from a generator of their own per batch, seeded by the run's seed
# and the batch's index within its corpus, so that the curve moves with the model rather than
# with the masks, and a corpus is scored under the same masks whichever mixture it is in. The
# stride keeps those seeds clear of the run's own.
VALIDATION_SEED_STRIDE = 1_000

# Where the run says what it wrote and what an epoch measured: a session on a platform without a
# tracking server keeps its log, and the log is then the only place a dropped run can be picked
# up from.
logger = logging.getLogger(__name__)


@dataclass
class _Best:
    """The epoch worth keeping so far: its mean relative validation and the weights it left."""

    relative: float
    weights: ArtifactRef


@dataclass
class _Session:
    """Everything one run holds while it runs; assembled before the first epoch, mutated by it."""

    configuration: ExperimentConfiguration
    mixture: TrainingMixture
    signature: RunSignature
    precision: TorchPrecision
    model: MaskedReconstruction
    loss: ReconstructionLoss
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    training: InterleavedLoader
    validation: tuple[tuple[TrainingCorpus, WindowLoader], ...]
    masking: TokenMasking
    draws: torch.Generator
    position: RunPosition
    started: float
    best: _Best | None = None


class TorchTrainingRuntime:
    """Trains the objective in this process, on whatever device the machine has.

    The loop is the ordinary one — a forward pass under autocast, gradients accumulated over as
    many micro-batches as the budget asks for, one optimiser step, one scheduler step — and what
    it adds is that it can be stopped and picked up, and that it keeps the epoch worth keeping.
    Checkpoints are written on a step boundary, when no gradient is pending, and carry the
    position so a resumed run skips exactly the micro-batches the interrupted one had consumed.
    Each corpus of the mixture is batched on its own and the batches are interleaved in the order
    the seed gives, in turns of as many batches as a step accumulates, so a batch pads to its own
    corpus's windows, a step is taken over one corpus, and the run weighs a corpus by its share
    of the steps. Every epoch scores each corpus's held-out side apart; an epoch whose mean
    relative loss over the corpora is the lowest so far writes its weights durably, and the
    backbone the run ends with is that epoch's, whatever the last epoch did.

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
            num_workers: Processes collating batches, for each corpus of the mixture; zero
                collates in this one.
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
        mixture: TrainingMixture,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        """Train the configuration over the mixture, an outcome per epoch.

        The run is assembled here and trained by the iterator, so a configuration this machine
        cannot honour, or a checkpoint from another run, is refused before anything is trained.

        Raises:
            IncompatibleCheckpointError: If the checkpoint belongs to another run, or to one that
                has finished.
            UnsupportedPrecisionError: If this device cannot train at the declared precision.
            DivergedRunError: While the run trains, if a loss or a gradient stops being finite.
        """
        return self._epochs(self._assemble(configuration, mixture, resume_from))

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
        mixture: TrainingMixture,
        resume_from: ArtifactRef | None,
    ) -> _Session:
        budget = configuration.budget
        precision = TorchPrecision(configuration.precision, self._device)
        signature = RunSignature.of(configuration, mixture)
        # Built on the host from the run's seed, so the same seed gives the same initial weights
        # whatever the device, and moved afterwards.
        torch.manual_seed(budget.seed)
        model = MaskedReconstruction(
            SetEncoder.for_vocabulary(
                configuration.architecture, mixture.vocabulary_size, dropout=configuration.dropout
            ),
            decoder_layers=configuration.decoder_layers,
            dropout=configuration.dropout,
        ).to(self._device)
        loss = ReconstructionLoss(configuration.loss)
        training = InterleavedLoader(
            [
                WindowLoader(
                    corpus.training,
                    batch_size=budget.batch_size,
                    seed=budget.seed,
                    num_workers=self._num_workers,
                )
                for corpus in mixture.corpora
            ],
            seed=budget.seed,
            group=budget.accumulation_steps,
        )
        validation = tuple(
            (
                corpus,
                WindowLoader(
                    corpus.validation, batch_size=budget.batch_size, seed=budget.seed, shuffle=False
                ),
            )
            for corpus in mixture.corpora
        )
        # No weight decay, said rather than inherited: a regularisation nobody chose should not be
        # one a reader has to look up. The day it varies it becomes a field of the experiment.
        optimiser = torch.optim.Adam(model.parameters(), lr=budget.learning_rate, weight_decay=0.0)
        session = _Session(
            configuration=configuration,
            mixture=mixture,
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
            started=time.perf_counter(),
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
        if checkpoint.best_relative is not None and checkpoint.best_weights is not None:
            session.best = _Best(checkpoint.best_relative, checkpoint.best_weights)

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
            outcome = EpochOutcome(
                epoch=epoch,
                training_loss=training,
                validation=validation,
                hidden_ratio=hidden_ratio,
                seconds=time.perf_counter() - started,
                checkpoint=checkpoint,
            )
            best = session.best
            if best is None or outcome.relative_validation < best.relative:
                best = _Best(outcome.relative_validation, self._write_model(session))
                session.best = best
                outcome = replace(outcome, weights=best.weights)
            if epoch + 1 == budget.epochs:
                outcome = replace(outcome, backbone=best.weights)
            # The last micro-batch of an epoch always closes a step, so a checkpoint the policy
            # makes due on that step is written here, once the epoch is scored: it then carries
            # this epoch's best, and a run picked up from it, which starts at the next epoch,
            # keeps what this one kept.
            if session.configuration.checkpoint.due_at(session.position.steps):
                outcome = replace(outcome, checkpoint=self._write_checkpoint(session))
            logger.info(
                "epoch %d of %d done in %.0f s, %.0f s into this session: training %.5f; %s",
                epoch + 1,
                budget.epochs,
                outcome.seconds,
                time.perf_counter() - session.started,
                training,
                "; ".join(
                    f"{scored.corpus} validation {scored.loss:.5f}, {scored.relative:.3f} of "
                    f"the trivial predictor's"
                    for scored in validation
                ),
            )
            yield outcome
            session.position = session.position.next_epoch()

    def _epoch(self, session: _Session, batches: int) -> tuple[float, ArtifactRef | None]:
        """One pass over the training windows from where the run stands.

        Returns the error over its hidden tokens and the last checkpoint written during it —
        which is what a run interrupted in the next epoch would be picked up from. A checkpoint
        due on the epoch's last step is not written here but once the epoch has been scored.
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
            if (
                stepped
                and index + 1 < batches
                and session.configuration.checkpoint.due_at(session.position.steps)
            ):
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

    def _validate(self, session: _Session) -> tuple[tuple[CorpusValidation, ...], float]:
        """What each corpus's held-out side cost this epoch, and the share of tokens scored.

        The masks are drawn per batch from a generator the seed fixes, so every epoch is scored
        under the same hidden tokens: what moves between epochs is the model. The channel-mean
        predictor is scored over the same tokens in the same pass, so that what a corpus is worth
        against nothing learnt is measured rather than looked up. Each corpus is scored on its
        own loader and reported on its own: a single number over the mixture would be the
        largest corpus's.
        """
        session.model.eval()
        scored_corpora: list[CorpusValidation] = []
        hidden_tokens_in_all, observed_in_all = 0, 0
        with torch.no_grad():
            for corpus, loader in session.validation:
                total, trivial, hidden_tokens = 0.0, 0.0, 0
                for index, on_host in enumerate(loader.batches_of(0)):
                    draws = torch.Generator().manual_seed(
                        session.configuration.budget.seed * VALIDATION_SEED_STRIDE + index
                    )
                    masks = session.masking.draw(on_host, draws)
                    batch, drawn = on_host.to(self._device), masks.to(self._device)
                    with session.precision.autocast():
                        predicted = session.model(batch, drawn)
                    error, counted = session.loss.summed(predicted, batch, drawn.hidden)
                    nothing_learnt, _ = session.loss.summed_over_mean(batch, drawn.hidden)
                    scored = int(counted)
                    total += float(error)
                    trivial += float(nothing_learnt)
                    hidden_tokens += scored
                    observed_in_all += int((~batch.padding_mask).sum())
                hidden_tokens_in_all += hidden_tokens
                scored_corpora.append(
                    CorpusValidation(
                        corpus=corpus.name,
                        tokens=hidden_tokens,
                        loss=total / max(hidden_tokens, 1),
                        trivial=trivial / max(hidden_tokens, 1),
                    )
                )
        return tuple(scored_corpora), hidden_tokens_in_all / max(observed_in_all, 1)

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
            best_relative=None if session.best is None else session.best.relative,
            best_weights=None if session.best is None else session.best.weights,
        )
        written = self._store.put(checkpoint.to_bytes(), Retention.TRANSIENT)
        logger.info(
            "checkpoint %s %s at step %d of epoch %d, %.0f s into this session",
            written.key,
            written.checksum,
            session.position.steps,
            session.position.epoch,
            time.perf_counter() - session.started,
        )
        return written

    def _write_model(self, session: _Session) -> ArtifactRef:
        """The weights the run was for, kept until someone removes them."""
        model = TrainedModel.of(
            session.configuration, session.mixture.vocabulary_size, session.model
        )
        return self._store.put(model.to_bytes(), Retention.DURABLE)
