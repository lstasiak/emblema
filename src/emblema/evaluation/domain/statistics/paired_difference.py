from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidPairedDifferenceError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind


@dataclass(frozen=True, kw_only=True)
class PairedDifference:
    """What a paired comparison found: the reduction, where it lies, and how surprising zero is.

    Invariants: the reduction is finite, and so is its share where the control's error gives
    one; the p-value lies in ``[0, 1]``.

    Attributes:
        reduction: How much lower the candidate's error is than the control's.
        relative_reduction: The reduction as a share of the control's error; ``None`` where the
            control made no error, so no share exists.
        interval: Where the reduction lies over resamples of the units.
        p_value: Two-sided: twice the smaller share of resamples on either side of zero, the
            observed reduction counted among them.
    """

    reduction: float
    relative_reduction: float | None
    interval: BootstrapInterval
    p_value: float

    def __post_init__(self) -> None:
        if not isfinite(self.reduction):
            raise InvalidPairedDifferenceError(f"reduction must be finite, got {self.reduction}")
        if self.relative_reduction is not None and not isfinite(self.relative_reduction):
            raise InvalidPairedDifferenceError(
                f"relative_reduction must be finite, got {self.relative_reduction}"
            )
        if not 0.0 <= self.p_value <= 1.0:
            raise InvalidPairedDifferenceError(f"p_value must lie in [0, 1], got {self.p_value}")

    def confirms(self, minimum: float, threshold: ThresholdKind) -> bool:
        """Whether the reduction is at least ``minimum``, its whole interval above zero.

        The minimum is a share of the control's error or an amount in its unit, as
        ``threshold`` states. A control that made no error leaves no share to reach.
        """
        match threshold:
            case ThresholdKind.RELATIVE:
                reached = self.relative_reduction is not None and self.relative_reduction >= minimum
            case ThresholdKind.ABSOLUTE:
                reached = self.reduction >= minimum
        return reached and self.interval.above_zero

    @property
    def distinguishable(self) -> bool:
        """Whether the interval keeps zero out, on either side."""
        return self.interval.excludes_zero
