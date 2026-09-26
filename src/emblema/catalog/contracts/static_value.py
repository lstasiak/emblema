from dataclasses import dataclass
from math import isfinite

from emblema.catalog.contracts.exceptions import InvalidObservedWindowError


@dataclass(frozen=True, kw_only=True)
class StaticValue:
    """One value that describes the whole window rather than an instant of it.

    Invariants: the channel is named, non-blank without surrounding whitespace; the value is
    finite.

    Attributes:
        channel: Name of the channel, as the corpus's vocabulary spells it; a timeless one.
        value: The value, in the channel's own unit, before any normalisation.
    """

    channel: str
    value: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidObservedWindowError(
                "a static value names its channel, non-blank without surrounding whitespace"
            )
        if not isfinite(self.value):
            raise InvalidObservedWindowError(f"value must be finite, got {self.value}")
