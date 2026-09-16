from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidUnitKeyError


@dataclass(frozen=True)
class UnitKey:
    """Identity of a unit a task is split and scored over: an engine, a machine, a stay.

    Evaluation names units in its own words even though the Catalog publishes the same strings:
    a published corpus lists its units as text, and a context that borrowed the Catalog's type
    would be reaching past the published language for a convenience.

    Attributes:
        value: Non-blank text without surrounding whitespace.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise InvalidUnitKeyError("unit key must be non-blank without surrounding whitespace")

    def __str__(self) -> str:
        return self.value
