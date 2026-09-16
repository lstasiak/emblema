from typing import Protocol

from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.identifiers import BackboneId


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

    def save(self, backbone: Backbone) -> None:
        """Store this state of the backbone, replacing the previous one."""
        ...
