from typing import Protocol

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.kernel.artifacts import ArtifactRef


class BackboneRepository(Protocol):
    """Persistence of the ``Backbone`` aggregate: whole backbones in, whole backbones out.

    The backbone is immutable, so ``save`` stores the state the caller holds and replaces
    whatever was stored under the same identity; one call is one transaction.
    """

    def get(self, backbone_id: BackboneId) -> Backbone:
        """The stored state of one backbone.

        Raises:
            BackboneNotFoundError: If no backbone has that identifier.
        """
        ...

    def get_by_artifact(self, artifact: ArtifactRef) -> Backbone:
        """The ready backbone whose weights are stored under ``artifact``.

        Weights are content-addressed, so the backbones holding one artifact are runs that
        delivered the same bytes — the same run ordered twice — and the one ordered first is
        the provenance; a run that starts from stored weights is traced to it this way.

        Raises:
            BackboneNotFoundError: If no ready backbone holds that artifact.
        """
        ...

    def save(self, backbone: Backbone) -> None:
        """Store this state of the backbone, replacing the previous one."""
        ...
