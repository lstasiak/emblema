import json
from collections.abc import Iterator
from math import ceil

from emblema.pretraining.domain.exceptions import IncompatibleCheckpointError
from emblema.pretraining.domain.training.corpus_validation import CorpusValidation
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.run_position import RunPosition
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention


class InMemoryTrainingRuntime:
    """A run without the arithmetic: the shape of a training run and none of the learning.

    It counts the epochs and the steps the configuration asks for over every corpus of the
    mixture, writes a checkpoint whenever the policy says one is due, writes the weights of every
    epoch because a loss that falls makes every epoch the best so far, and reports the last
    epoch's as the backbone, which under a falling loss is the best epoch's. The loss falls
    because a curve that falls is what callers of the port are written against. Whoever wants a
    model that learnt something asks the runtime that trains; whoever wants to know that a caller
    logs every epoch, resumes where it stopped and keeps the epoch the run kept does not need
    one, and does not need torch either.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def train(
        self,
        configuration: ExperimentConfiguration,
        mixture: TrainingMixture,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        signature = RunSignature.of(configuration, mixture)
        batches = sum(
            ceil(len(corpus.training) / configuration.budget.batch_size)
            for corpus in mixture.corpora
        )
        position = (
            RunPosition.start()
            if resume_from is None
            else self._resumed(resume_from, signature, configuration, batches)
        )
        return self._epochs(configuration, mixture, signature, position, batches)

    def _epochs(
        self,
        configuration: ExperimentConfiguration,
        mixture: TrainingMixture,
        signature: RunSignature,
        position: RunPosition,
        batches: int,
    ) -> Iterator[EpochOutcome]:
        budget = configuration.budget
        while position.epoch < budget.epochs:
            if position.batches >= batches:
                position = position.next_epoch()
                continue
            checkpoint = None
            for index in range(position.batches, batches):
                stepped = budget.takes_a_step_at(index, batches)
                position = position.after_batch(stepped=stepped)
                if stepped and configuration.checkpoint.due_at(position.steps):
                    checkpoint = self._write(signature, position, Retention.TRANSIENT)
            final = position.epoch + 1 == budget.epochs
            # A loss that falls with the epoch and nothing more: the numbers are a shape, not a
            # measurement, and calling them one anywhere would be a lie. The trivial predictor is
            # given a loss of one for the same reason, so that the relative loss is the shape too.
            falling = 1.0 / (position.epoch + 1)
            weights = self._model(signature, mixture, position.epoch)
            yield EpochOutcome(
                epoch=position.epoch,
                training_loss=falling,
                validation=tuple(
                    CorpusValidation(
                        corpus=corpus.name,
                        tokens=len(corpus.validation),
                        loss=falling,
                        trivial=1.0,
                    )
                    for corpus in mixture.corpora
                ),
                hidden_ratio=configuration.masking.expected_ratio,
                seconds=0.0,
                checkpoint=checkpoint,
                weights=weights,
                backbone=weights if final else None,
            )
            position = position.next_epoch()

    def _resumed(
        self,
        ref: ArtifactRef,
        signature: RunSignature,
        configuration: ExperimentConfiguration,
        batches: int,
    ) -> RunPosition:
        """Where the checkpoint says the run stood, refused where it belongs to another run.

        Raises:
            IncompatibleCheckpointError: If the checkpoint is not this run's, or its run is over.
        """
        try:
            stored = json.loads(self._store.get(ref))
            found = RunSignature(stored["signature"])
            position = RunPosition(
                epoch=stored["epoch"], batches=stored["batches"], steps=stored["steps"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(f"not a checkpoint this can read: {error}") from error
        if found != signature:
            raise IncompatibleCheckpointError(
                f"checkpoint of run {found} offered to run {signature}"
            )
        epochs = configuration.budget.epochs
        if position.epoch >= epochs or (
            position.epoch == epochs - 1 and position.batches >= batches
        ):
            raise IncompatibleCheckpointError("the run this checkpoint comes from has finished")
        return position

    def _write(
        self, signature: RunSignature, position: RunPosition, retention: Retention
    ) -> ArtifactRef:
        state = {
            "signature": signature.digest,
            "epoch": position.epoch,
            "batches": position.batches,
            "steps": position.steps,
        }
        return self._store.put(json.dumps(state, sort_keys=True).encode(), retention)

    def _model(self, signature: RunSignature, mixture: TrainingMixture, epoch: int) -> ArtifactRef:
        stated = {"signature": signature.digest, "corpora": mixture.name, "epoch": epoch}
        return self._store.put(json.dumps(stated, sort_keys=True).encode(), Retention.DURABLE)
