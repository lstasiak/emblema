from dataclasses import dataclass

from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True)
class Adapters:
    """Every port implementation the process runs on, so that what a use case got can be read."""

    store: ArtifactStore
    promotables: PromotableArtifactRepository
    served: ServedModelRepository
    clock: Clock
    ids: IdGenerator
