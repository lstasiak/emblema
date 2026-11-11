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

    def bounds_of(self, channel: str) -> ValueBounds | None:
        """The bounds of ``channel``; ``None`` where the policy leaves it as it is."""
        return next((each for each in self.bounds if each.channel == channel), None)

    def cleaned_observation(self, observation: Observation) -> Observation | None:
        """The observation as the policy allows it; ``None`` where its value is removed."""
        value = self._cleaned(observation.channel, observation.value)
        return None if value is None else replace(observation, value=value)

    def cleaned_feature(self, feature: StaticFeature) -> StaticFeature | None:
        """The descriptor as the policy allows it; ``None`` where its value is removed."""
        value = self._cleaned(feature.channel, feature.value)
        return None if value is None else replace(feature, value=value)

    def _cleaned(self, channel: str, value: float) -> float | None:
        bounds = self.bounds_of(channel)
        return value if bounds is None else bounds.cleaned(value)
