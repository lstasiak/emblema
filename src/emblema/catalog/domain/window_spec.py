import math
from collections.abc import Iterator
from dataclasses import dataclass

from emblema.catalog.domain.corpus_unit import TimeExtent
from emblema.catalog.domain.exceptions import InvalidWindowSpecError


@dataclass(frozen=True)
class WindowSpec:
    """How a unit's time axis is cut into windows: a length and a stride, in the unit's time unit.

    Windows are laid from the start of a unit's extent, every ``stride``, and only a window that
    fits inside the extent counts; the tail that no full window covers is left out. Both numbers
    are elapsed time, never a count of samples, so the same rule serves a corpus sampled every
    cycle and one sampled whenever a sensor reports. Scale lives in configuration: there is no
    default window.

    Attributes:
        length: Extent of one window; positive, finite.
        stride: Distance between the starts of consecutive windows; positive, finite.
    """

    length: float
    stride: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.length) and self.length > 0):
            raise InvalidWindowSpecError(f"length must be positive and finite, got {self.length}")
        if not (math.isfinite(self.stride) and self.stride > 0):
            raise InvalidWindowSpecError(f"stride must be positive and finite, got {self.stride}")

    def windows_over(self, extent: TimeExtent) -> Iterator[TimeExtent]:
        """Every window that fits inside ``extent``, in time order."""
        index = 0
        while True:
            start = extent.start + index * self.stride
            end = start + self.length
            if end > extent.end:
                return
            yield TimeExtent(start, end)
            index += 1
