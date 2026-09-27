from collections import Counter
from dataclasses import dataclass
from math import isfinite

from emblema.catalog.contracts.exceptions import InvalidObservedWindowError
from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.static_value import StaticValue


@dataclass(frozen=True, kw_only=True)
class ObservedWindow:
    """A span of a caller's readings, stated in raw values, to be tokenised as a corpus was.

    The window says where it starts and how long it is, because a token's time is its position
    within the window as a share of the length: readings alone would put the first of them at
    the start of every window and lose where it really fell. Readings may arrive in any order;
    they are placed by their time.

    Invariants: the start is finite and the length positive and finite; every reading falls
    inside ``[start, start + length)``; at least one reading is timed, because a window of
    static values alone holds no measurement; no static channel is stated twice.

    Attributes:
        start: First instant of the window on the caller's time axis.
        length: Extent of the window, in the same unit.
        observations: Timed readings, any order.
        static_features: Values that hold for the whole window, one per channel.
    """

    start: float
    length: float
    observations: tuple[ObservedValue, ...]
    static_features: tuple[StaticValue, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.start):
            raise InvalidObservedWindowError(f"start must be finite, got {self.start}")
        if not isfinite(self.length) or self.length <= 0.0:
            raise InvalidObservedWindowError(
                f"length must be positive and finite, got {self.length}"
            )
        if not self.observations:
            raise InvalidObservedWindowError("a window holds at least one timed reading")
        end = self.start + self.length
        for reading in self.observations:
            if not self.start <= reading.time < end:
                raise InvalidObservedWindowError(
                    f"a reading of {reading.channel!r} at {reading.time} lies outside "
                    f"[{self.start}, {end})"
                )
        counts = Counter(feature.channel for feature in self.static_features)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise InvalidObservedWindowError(f"static channels stated twice: {duplicates}")

    @property
    def end(self) -> float:
        return self.start + self.length

    def channels(self) -> frozenset[str]:
        """Every channel the window has a reading on, timed or static."""
        return frozenset(reading.channel for reading in self.observations) | frozenset(
            feature.channel for feature in self.static_features
        )
