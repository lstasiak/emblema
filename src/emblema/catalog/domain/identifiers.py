"""Identifiers private to the Data Catalog."""

from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidUnitKeyError
from emblema.shared.kernel.identifiers import EntityId

_PART_SEPARATOR = "/"


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
    see which part a unit came from. How the key states the part is the key's alone: readers
    compose one with ``within`` and take it apart through ``part`` and ``name``.

    Attributes:
        value: Non-blank text without surrounding whitespace.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise InvalidUnitKeyError("unit key must be non-blank without surrounding whitespace")

    @classmethod
    def within(cls, part: str, name: str) -> "UnitKey":
        """The key of the unit a corpus calls ``name`` in the part it calls ``part``.

        Raises:
            InvalidUnitKeyError: If either is blank or has surrounding whitespace, or the part
                holds the separator, which would leave the key readable as another part's.
        """
        for text, what in ((part, "part"), (name, "name")):
            if not text or text != text.strip():
                raise InvalidUnitKeyError(
                    f"a unit's {what} must be non-blank without surrounding whitespace"
                )
        if _PART_SEPARATOR in part:
            raise InvalidUnitKeyError(f"a part must not hold {_PART_SEPARATOR!r}, got {part!r}")
        return cls(f"{part}{_PART_SEPARATOR}{name}")

    @property
    def part(self) -> str | None:
        """The part of the corpus the unit was read from; ``None`` where the key names no part."""
        part, separator, _ = self.value.partition(_PART_SEPARATOR)
        return part if separator else None

    @property
    def name(self) -> str:
        """What the corpus calls the unit within its part; the whole key where there is no part."""
        _, separator, name = self.value.partition(_PART_SEPARATOR)
        return name if separator else self.value

    def belongs_to(self, part: str) -> bool:
        """Whether this unit was read from that part of its corpus."""
        return self.part == part

    def __str__(self) -> str:
        return self.value
