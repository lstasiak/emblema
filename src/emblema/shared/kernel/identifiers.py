from dataclasses import dataclass
from typing import Self
from uuid import UUID

from emblema.shared.kernel.exceptions import InvalidEntityIdError


@dataclass(frozen=True)
class EntityId:
    """Identity of an entity, wrapped so that identifiers of different kinds never mix.

    Each bounded context subclasses this for every aggregate or entity it identifies
    (``class CorpusId(EntityId)``). Equality is per class: two identifiers of different kinds
    built from the same UUID are not equal.

    Attributes:
        value: The underlying UUID.
    """

    value: UUID

    @classmethod
    def parse(cls, text: str) -> Self:
        """Build an identifier from its canonical string form.

        Raises:
            InvalidEntityIdError: If ``text`` is not a valid UUID.
        """
        try:
            value = UUID(text)
        except ValueError as error:
            raise InvalidEntityIdError(f"not a UUID: {text!r}") from error
        return cls(value)

    def __str__(self) -> str:
        return str(self.value)
