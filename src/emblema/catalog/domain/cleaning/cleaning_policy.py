from dataclasses import dataclass, replace

from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.exceptions import InvalidCleaningPolicyError
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature


@dataclass(frozen=True)
class CleaningPolicy:
    """The bounds a publication holds a corpus's values to, one per channel it names.

    A channel the policy does not name is left as it is. The policy is applied to the values
    that fit the statistics and to the values that are tokenised alike, timed or timeless, so a
    model never sees a value the statistics did not.

    Invariants: at least one channel is named, and none twice.

    Attributes:
        bounds: The bounds of each channel named.
    """

    bounds: tuple[ValueBounds, ...]

    def __post_init__(self) -> None:
        if not self.bounds:
            raise InvalidCleaningPolicyError("a cleaning policy names at least one channel")
        channels = [each.channel for each in self.bounds]
        repeated = sorted({channel for channel in channels if channels.count(channel) > 1})
        if repeated:
            raise InvalidCleaningPolicyError(f"channels bounded twice: {repeated}")

    @property
    def channels(self) -> frozenset[str]:
        return frozenset(each.channel for each in self.bounds)

    def cleaned[T: (Observation, StaticFeature)](self, reading: T) -> T | None:
        """The reading, timed or timeless, as the policy allows it; ``None`` where it is removed."""
        bounds = next((each for each in self.bounds if each.channel == reading.channel), None)
        if bounds is None:
            return reading
        value = bounds.cleaned(reading.value)
        return None if value is None else replace(reading, value=value)
