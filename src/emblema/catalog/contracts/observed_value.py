from dataclasses import dataclass
from math import isfinite

from emblema.catalog.contracts.exceptions import InvalidObservedWindowError


@dataclass(frozen=True, kw_only=True)
class ObservedValue:
    """One reading of one channel at one instant, as a caller outside the Catalog states it.

    Time is a point on the caller's own axis, in whatever unit the corpus the model was fitted
    to measures it in; only positions within the window matter, so its origin does not.

    Invariants: the channel is named, non-blank without surrounding whitespace; time and value
    are finite.

    Attributes:
        channel: Name of the channel, as the corpus's vocabulary spells it.
        time: When it was observed.
        value: What was observed, in the channel's own unit, before any normalisation.
    """

    channel: str
    time: float
    value: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidObservedWindowError(
                "a reading names its channel, non-blank without surrounding whitespace"
            )
        if not isfinite(self.time):
            raise InvalidObservedWindowError(f"time must be finite, got {self.time}")
        if not isfinite(self.value):
            raise InvalidObservedWindowError(f"value must be finite, got {self.value}")
