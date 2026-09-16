"""Both machines of a handoff over in-memory adapters, sharing one store and one exchange.

The machine that orders and accepts and the machine that trains are the same process here, but
they meet only where they would in production: in the store the weights land in and in the
exchange the documents pass through. Every tracker built is kept, so a test can read what each
run recorded.
"""

from typing import Any

from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.backbone_repository import InMemoryBackboneRepository
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.handoff_exchange import InMemoryHandoffExchange
from emblema.pretraining.adapters.in_memory.training_corpus_reader import (
    InMemoryTrainingCorpusReader,
)
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResult,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrder,
)
from emblema.pretraining.application.use_cases.order_pretraining import (
    OrderPretraining,
    OrderPretrainingCommand,
)
from emblema.pretraining.application.use_cases.pretrain_backbone import PretrainBackbone
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.support.handoff import COMMIT, CONFIGURATION, CORPUS, MANIFEST, ORDERED_AT
from tests.support.handoff import pretraining_input as described


class InMemoryHandoff:
    """The adapters both machines share, and the use cases each of them runs."""

    def __init__(self) -> None:
        self.store = InMemoryArtifactStore()
        self.exchange = InMemoryHandoffExchange()
        self.backbones = InMemoryBackboneRepository()
        self.reader = InMemoryTrainingCorpusReader()
        self.reader.publish(MANIFEST, described(), CORPUS)
        self.ids = SequentialIdGenerator()
        self.clock = FixedClock(ORDERED_AT)
        self.trackers: list[InMemoryExperimentTracker] = []

    def order(self) -> OrderPretraining:
        return OrderPretraining(self.reader, self.backbones, self.exchange, self.ids, self.clock)

    def fulfil(self) -> FulfilPretrainingOrder:
        pretrain = PretrainBackbone(InMemoryTrainingRuntime(self.store), self._tracker())
        return FulfilPretrainingOrder(self.exchange, self.reader, pretrain)

    def accept(self, result: ArtifactRef) -> AcceptPretrainingResult:
        runtime = HandoffTrainingRuntime(self.exchange, self.store, result)
        pretrain = PretrainBackbone(runtime, self._tracker())
        return AcceptPretrainingResult(
            self.exchange, self.reader, self.backbones, pretrain, self.clock
        )

    def _tracker(self) -> InMemoryExperimentTracker:
        tracker = InMemoryExperimentTracker()
        self.trackers.append(tracker)
        return tracker


def order_command(**overrides: Any) -> OrderPretrainingCommand:
    stated: dict[str, Any] = {
        "configuration": CONFIGURATION,
        "corpus": CORPUS.name,
        "manifest": MANIFEST,
        "run": "first",
        "git_commit": COMMIT,
    }
    return OrderPretrainingCommand(**(stated | overrides))
