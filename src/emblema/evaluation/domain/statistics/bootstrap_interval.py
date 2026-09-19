from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidBootstrapIntervalError


@dataclass(frozen=True, kw_only=True)
class BootstrapInterval:
    """Where a resampled quantity lies with a stated confidence.

    Invariants: both ends are finite and the low end does not exceed the high one; the level lies
    strictly between zero and one.

    Attributes:
        low: The lower end.
        high: The upper end.
        level: The confidence the interval is stated at, as a share.
    """

    low: float
    high: float
    level: float

    def __post_init__(self) -> None:
        if not (isfinite(self.low) and isfinite(self.high)) or self.low > self.high:
            raise InvalidBootstrapIntervalError(
                f"an interval needs finite ends in order, got [{self.low}, {self.high}]"
            )
        if not 0.0 < self.level < 1.0:
            raise InvalidBootstrapIntervalError(f"level must lie in (0, 1), got {self.level}")

    @property
    def above_zero(self) -> bool:
        """Whether the whole interval lies above zero — the claim's form of a positive result."""
        return self.low > 0.0

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def within(self, margin: float) -> bool:
        """Whether the whole interval lies inside ``±margin`` — the form of an equivalence claim."""
        return -margin <= self.low and self.high <= margin

    def __str__(self) -> str:
        return f"[{self.low:+.3f}, {self.high:+.3f}]"
