import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from emblema.catalog.domain.exceptions import InvalidWindowSpecError
from emblema.catalog.domain.measurements.time_extent import TimeExtent


class ShortUnits(StrEnum):
    """What a unit shorter than one window yields.

    A window is laid only where it fits, so a unit shorter than the window yields nothing and
    is silent in the published corpus. Where the units are records of varying length and their
    length is part of what they say, such as telemetry fragments cut by their operators, the
    unit yields one window of the nominal length anchored at its start, so every unit speaks and
    its duration is read off how much of the window it fills.

    Attributes:
        DROPPED: A unit shorter than a window yields no window.
        ONE_WINDOW: A unit shorter than a window yields one window of the nominal length,
            starting with the unit; the unit's readings fill its beginning.
    """

    DROPPED = "dropped"
    ONE_WINDOW = "one_window"


@dataclass(frozen=True)
class WindowSpec:
    """How a unit's time axis is cut into windows: a length and a stride, in the unit's time unit.

    Windows are laid from the start of a unit's extent, every ``stride``, and only a window that
    fits inside the extent counts; the tail that no full window covers is left out, and what a
    unit too short for one window yields is the policy's business. Both numbers are elapsed
    time, never a count of samples, so the same rule serves a corpus sampled every cycle and one
    sampled whenever a sensor reports. Scale lives in configuration: there is no default window.

    Attributes:
        length: Extent of one window; positive, finite.
        stride: Distance between the starts of consecutive windows; positive, finite.
        short_units: What a unit shorter than one window yields.
    """

    length: float
    stride: float
    short_units: ShortUnits = ShortUnits.DROPPED

    def __post_init__(self) -> None:
        if not (math.isfinite(self.length) and self.length > 0):
            raise InvalidWindowSpecError(f"length must be positive and finite, got {self.length}")
        if not (math.isfinite(self.stride) and self.stride > 0):
            raise InvalidWindowSpecError(f"stride must be positive and finite, got {self.stride}")

    def windows_over(self, extent: TimeExtent) -> Iterator[TimeExtent]:
        """Every window that fits inside ``extent``, in time order.

        A unit shorter than one window yields one window of the nominal length under the policy
        that says so, which then reaches past the unit's end; its readings lie in its beginning.
        """
        if extent.length < self.length:
            if self.short_units is ShortUnits.ONE_WINDOW:
                yield TimeExtent(extent.start, extent.start + self.length)
            return
        index = 0
        while True:
            start = extent.start + index * self.stride
            end = start + self.length
            if end > extent.end:
                return
            yield TimeExtent(start, end)
            index += 1
