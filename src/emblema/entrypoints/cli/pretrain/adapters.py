from dataclasses import dataclass

from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True)
class Adapters:
    """Every port implementation the process runs on, so that what a use case got can be read.

    The registry is ``None`` in a process that has no database to keep one in — the machine that
    only trains — and the use cases that need it are absent from its services.
    """

    store: ArtifactStore
    backbones: BackboneRepository | None
    reader: TrainingCorpusReader
    exchange: HandoffExchange
    runtime: TrainingRuntime
    tracker: ExperimentTracker
    clock: Clock
    ids: IdGenerator
