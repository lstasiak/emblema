from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidUncertaintyDecompositionError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval


@dataclass(frozen=True, kw_only=True)
class UncertaintyDecomposition:
    """How much of a comparison's uncertainty the units account for, and how much the repeats add.

    Two intervals around one reduction: the registered one, over resamples of the units alone
    with the repeats pooled, and one over resamples of the repeats and of the units both. The
    second is never narrower. Their difference in variance, as a share of the second, is how
    much of the uncertainty comes from initialisation and the draw of labels rather than from
    which units happened to be held out. This is an ablation of the method, not a verdict: the
    registered rules read the first interval, and this says what they leave out.

    Invariants: both intervals are stated at one level; the share lies in ``[0, 1]``.

    Attributes:
        over_units: The interval over resamples of the units, repeats pooled.
        over_units_and_repeats: The interval over resamples of the repeats and the units.
        share_from_repeats: The share of the two-level variance the units alone do not carry.
    """

    over_units: BootstrapInterval
    over_units_and_repeats: BootstrapInterval
    share_from_repeats: float

    def __post_init__(self) -> None:
        if self.over_units.level != self.over_units_and_repeats.level:
            raise InvalidUncertaintyDecompositionError(
                "both intervals of a decomposition are stated at one level, got "
                f"{self.over_units.level} and {self.over_units_and_repeats.level}"
            )
        if not isfinite(self.share_from_repeats) or not 0.0 <= self.share_from_repeats <= 1.0:
            raise InvalidUncertaintyDecompositionError(
                f"share_from_repeats must lie in [0, 1], got {self.share_from_repeats}"
            )

    @property
    def widening(self) -> float:
        """How much wider the two-level interval is than the registered one, in the error's unit."""
        return (self.over_units_and_repeats.high - self.over_units_and_repeats.low) - (
            self.over_units.high - self.over_units.low
        )
