"""Identifiers private to the Data Catalog."""

from dataclasses import dataclass

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

    Attributes:
        value: Non-blank text without surrounding whitespace.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise InvalidUnitKeyError("unit key must be non-blank without surrounding whitespace")

    def __str__(self) -> str:
        return self.value
