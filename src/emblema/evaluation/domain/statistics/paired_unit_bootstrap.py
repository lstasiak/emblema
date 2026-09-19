import random
from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidPairedUnitBootstrapError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors


@dataclass(frozen=True, kw_only=True)
class PairedUnitBootstrap:
    """Resamples the units of a paired comparison, and says where their difference lies.

    Windows of one unit are not independent — they overlap, and a unit is one realisation of
    the process — so the unit is the level resampled, with replacement, the same units on both
    sides of every resample. The interval is the percentile one over the resampled reductions;
    the p-value is two-sided, twice the smaller share of resamples on either side of zero, which
    is the test the percentile interval inverts; the share counts the observed reduction as one
    more resample on its side, so no p-value is zero and the smallest one says how many
    resamples were drawn. Seeded, so one comparison always gets the same interval, and stated in
    plain Python: the sums are per unit and the units are few.

    Invariants: at least one resample; the level lies strictly between zero and one.

    Attributes:
        resamples: How many resamples the interval is read off.
        seed: Seed of the resampling.
        level: The confidence the interval is stated at, as a share.
    """

    resamples: int = 10_000
    seed: int = 1
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.resamples < 1:
            raise InvalidPairedUnitBootstrapError(
                f"resamples must be positive, got {self.resamples}"
            )
        if not 0.0 < self.level < 1.0:
            raise InvalidPairedUnitBootstrapError(f"level must lie in (0, 1), got {self.level}")

    def compare(self, paired: PairedUnitErrors) -> PairedDifference:
        """The reduction the candidate makes over the control, with its interval and p-value."""
        count = len(paired.control)
        draws = random.Random(self.seed)
        reductions = sorted(
            paired.reduction_over(draws.choices(range(count), k=count))
            for _ in range(self.resamples)
        )
        tail = (1.0 - self.level) / 2.0
        below = (1 + sum(1 for reduction in reductions if reduction <= 0.0)) / (self.resamples + 1)
        above = (1 + sum(1 for reduction in reductions if reduction >= 0.0)) / (self.resamples + 1)
        return PairedDifference(
            reduction=paired.reduction,
            relative_reduction=paired.relative_reduction,
            interval=BootstrapInterval(
                low=_quantile(reductions, tail),
                high=_quantile(reductions, 1.0 - tail),
                level=self.level,
            ),
            p_value=min(1.0, 2.0 * min(below, above)),
        )


def _quantile(ascending: Sequence[float], share: float) -> float:
    """The value ``share`` of the way through ``ascending``, interpolated between neighbours."""
    position = share * (len(ascending) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ascending) - 1)
    weight = position - lower
    return ascending[lower] * (1.0 - weight) + ascending[upper] * weight
