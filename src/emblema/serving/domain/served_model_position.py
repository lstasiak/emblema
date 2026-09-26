from dataclasses import dataclass
from typing import Self

from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class ServedModelPosition:
    """Where a served model stands in the order its list is read in: promotion, then identity.

    A page continues after a position rather than after a row, so the next page needs nothing
    stored: a model withdrawn or promoted between two pages leaves the position where it was.

    Attributes:
        promoted_at: When the model was promoted; later ones are listed first.
        served_model_id: Identity, which orders models promoted at the same instant.
    """

    promoted_at: UtcDateTime
    served_model_id: ServedModelId

    @classmethod
    def of(cls, model: ServedModel) -> Self:
        return cls(promoted_at=model.promoted_at, served_model_id=model.served_model_id)
