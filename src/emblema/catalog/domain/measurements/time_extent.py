import math
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidTimeExtentError


@dataclass(frozen=True)
class TimeExtent:
    """A half-open span ``[start, end)`` on a unit's time axis.

    The extent of a unit is a fact the reader states, not the span of its observations: a hospital
    stay covers its protocol's 48 hours whether or not the last hour holds a measurement, and an
    engine of ``L`` cycles spans ``[1, L + 1)`` because a cycle occupies the unit interval that
    starts at its index. Windows are laid over the extent, so it decides how many there are.

    Attributes:
        start: First instant inside the span; finite.
        end: First instant beyond the span; finite, after ``start``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise InvalidTimeExtentError("start and end must be finite")
        if self.start >= self.end:
            raise InvalidTimeExtentError(f"start must precede end, got [{self.start}, {self.end})")

    @property
    def length(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        return self.start <= time < self.end
