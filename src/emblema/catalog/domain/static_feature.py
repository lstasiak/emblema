import math
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidStaticFeatureError


@dataclass(frozen=True)
class StaticFeature:
    """One value that describes a whole unit rather than an instant of it.

    A patient's age or a device's rated power holds for every window of the unit and has no
    time; it becomes a timeless token in each of them. The channel is a vocabulary entry like any
    other, declared timeless in the corpus schema.

    Attributes:
        channel: Name of the channel within its corpus, non-blank.
        value: Raw value as read; finite.
    """

    channel: str
    value: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidStaticFeatureError(
                "channel name must be non-blank without surrounding whitespace"
            )
        if not math.isfinite(self.value):
            raise InvalidStaticFeatureError(f"value must be finite, got {self.value}")
