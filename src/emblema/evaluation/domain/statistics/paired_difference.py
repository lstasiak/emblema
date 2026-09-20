from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidPairedDifferenceError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval


@dataclass(frozen=True, kw_only=True)
class PairedDifference:
    """What a paired comparison found: the reduction, where it lies, and how surprising zero is.

    Invariants: the reduction and its share are finite; the p-value lies in ``[0, 1]``.

    Attributes:
        reduction: How much lower the candidate's error is than the control's.
        relative_reduction: The reduction as a share of the control's error.
        interval: Where the reduction lies over resamples of the units.
        p_value: Two-sided: twice the smaller share of resamples on either side of zero, the
            observed reduction counted among them.
    """

    reduction: float
    relative_reduction: float
    interval: BootstrapInterval
    p_value: float

    def __post_init__(self) -> None:
        for label, value in (
            ("reduction", self.reduction),
            ("relative_reduction", self.relative_reduction),
        ):
            if not isfinite(value):
                raise InvalidPairedDifferenceError(f"{label} must be finite, got {value}")
        if not 0.0 <= self.p_value <= 1.0:
            raise InvalidPairedDifferenceError(f"p_value must lie in [0, 1], got {self.p_value}")

    def confirms(self, minimum_relative_reduction: float) -> bool:
        """Whether the reduction is at least the share asked, its whole interval above zero."""
        return self.relative_reduction >= minimum_relative_reduction and self.interval.above_zero

    @property
    def distinguishable(self) -> bool:
        """Whether the interval keeps zero out, on either side."""
        return self.interval.excludes_zero
