"""Identifiers private to the Data Catalog."""

from dataclasses import dataclass
from typing import ClassVar

from emblema.catalog.domain.exceptions import InvalidUnitKeyError
from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class CorpusId(EntityId):
    """Identity of a corpus, private to the Catalog.

    Other contexts reference versions, never the corpus, so this identifier stays in the domain
    while ``CorpusVersionId`` lives in the published language.
    """


@dataclass(frozen=True)
class UnitKey:
    """Identity of a unit within its corpus, as the reader names it: an engine, a machine, a stay.

    The key is the reader's, not a generated identity, because it must name the same unit across
    readings of the same data and be recognisable in the source.

    A corpus that arrives in parts — challenge sets, missions, folders — names its units within
    them, so that two units its publishers numbered alike stay apart and a reader of the key can
    see which part a unit came from.

    Attributes:
        value: Non-blank text without surrounding whitespace.
    """

    value: str
    # How a key states the part a unit belongs to, ahead of the name the corpus gives the unit.
    PART_SEPARATOR: ClassVar[str] = "/"

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise InvalidUnitKeyError("unit key must be non-blank without surrounding whitespace")

    @classmethod
    def within(cls, part: str, name: str) -> "UnitKey":
        """The key of the unit a corpus calls ``name`` in the part it calls ``part``.

        Raises:
            InvalidUnitKeyError: If either is blank or has surrounding whitespace.
        """
        for text, what in ((part, "part"), (name, "name")):
            if not text or text != text.strip():
                raise InvalidUnitKeyError(
                    f"a unit's {what} must be non-blank without surrounding whitespace"
                )
        return cls(f"{part}{cls.PART_SEPARATOR}{name}")

    def belongs_to(self, part: str) -> bool:
        """Whether this unit was read from that part of its corpus."""
        return self.value.startswith(f"{part}{self.PART_SEPARATOR}")

    def __str__(self) -> str:
        return self.value
