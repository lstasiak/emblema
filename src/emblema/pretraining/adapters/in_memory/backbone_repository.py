from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import BackboneNotFoundError
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.kernel.artifacts import ArtifactRef


class InMemoryBackboneRepository:
    """Repository over a dictionary: the fake of the port for application tests."""

    def __init__(self) -> None:
        self._backbones: dict[BackboneId, Backbone] = {}

    def get(self, backbone_id: BackboneId) -> Backbone:
        try:
            return self._backbones[backbone_id]
        except KeyError:
            raise BackboneNotFoundError(f"no backbone {backbone_id}") from None

    def get_by_artifact(self, artifact: ArtifactRef) -> Backbone:
        holding = sorted(
            (backbone for backbone in self._backbones.values() if backbone.artifact == artifact),
            key=lambda backbone: backbone.ordered_at.value,
        )
        if not holding:
            raise BackboneNotFoundError(f"no ready backbone holds {artifact.key}")
        return holding[0]

    def save(self, backbone: Backbone) -> None:
        self._backbones[backbone.id] = backbone
