from dataclasses import dataclass
from typing import Self
from uuid import UUID


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
            ValueError: If ``text`` is not a valid UUID.
        """
        return cls(UUID(text))

    def __str__(self) -> str:
        return str(self.value)
