"""The handoff runtime driven the way the port's contract drives every runtime.

Training "elsewhere" is the in-memory runtime over the same store; what it reports goes through
an exchange as a result, and the handoff runtime replays it. Lazy like the runtimes it stands
beside: the other machine trains when the first epoch is asked for, so nothing is written before.
"""

from collections.abc import Iterator

from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.handoff_exchange import InMemoryHandoffExchange
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.support.handoff import COMMIT, ORDER_REF, backbone_id


class SimulatedPlatform:
    """A runtime whose training happens on the other side of a handoff exchange."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self.exchange = InMemoryHandoffExchange()

    def train(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        epochs = list(
            InMemoryTrainingRuntime(self._store).train(configuration, corpus, resume_from)
        )
        kept = epochs[-1].backbone
        assert kept is not None, "the in-memory runtime keeps the weights on the last epoch"
        reported = self.exchange.report(
            PretrainingResult(
                order=ORDER_REF,
                backbone=backbone_id(),
                configuration=configuration,
                corpus=corpus.shape,
                git_commit=COMMIT,
                resumed_from=resume_from,
                outcome=TrainingOutcome(backbone=kept, epochs=tuple(epochs)),
            )
        )
        replay = HandoffTrainingRuntime(self.exchange, self._store, reported)
        yield from replay.train(configuration, corpus, resume_from)
