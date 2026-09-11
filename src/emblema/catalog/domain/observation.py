import math
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidObservationError


@dataclass(frozen=True)
class Observation:
    """One value of one channel at one instant, as the corpus reader delivers it.

    Time is a point on the time axis of the unit the observation belongs to, in whatever unit the
    reader documents for its corpus (a cycle, an hour, a second); the tokenizer only ever uses
    differences and positions within a window, so the origin of the axis carries no meaning. A
    value that is missing in the source is no observation at all, so every value here is finite.

    Attributes:
        channel: Name of the channel within its corpus, non-blank.
        time: Instant on the unit's time axis; finite.
        value: Raw value as read; finite.
    """

    channel: str
    time: float
    value: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidObservationError(
                "channel name must be non-blank without surrounding whitespace"
            )
        if not math.isfinite(self.time):
            raise InvalidObservationError(f"time must be finite, got {self.time}")
        if not math.isfinite(self.value):
            raise InvalidObservationError(f"value must be finite, got {self.value}")
