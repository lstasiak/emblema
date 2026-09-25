import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import pvariance

from emblema.evaluation.domain.exceptions import InvalidTwoLevelBootstrapError
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.uncertainty_decomposition import (
    UncertaintyDecomposition,
)


@dataclass(frozen=True, kw_only=True)
class TwoLevelBootstrap:
    """Resamples the repeats of a comparison as well as its units, to see what the units leave out.

    The registered interval resamples units with the repeats pooled, on the grounds that a
    handful of repeats is not an interval. That is a decision about what the interval claims, not
    a claim that initialisation adds nothing; this procedure measures what it adds, by drawing
    repeats and units with replacement in every resample and setting that spread beside the
    units-only one. With a few repeats the outer level is coarse, which is why this is an
    ablation of the method and never the reading itself.

    Invariants: at least one resample; the level lies strictly between zero and one.

    Attributes:
        resamples: How many resamples each interval is read off.
        seed: Seed of the resampling.
        level: The confidence both intervals are stated at, as a share.
    """

    resamples: int = 10_000
    seed: int = 1
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.resamples < 1:
            raise InvalidTwoLevelBootstrapError(f"resamples must be positive, got {self.resamples}")
        if not 0.0 < self.level < 1.0:
            raise InvalidTwoLevelBootstrapError(f"level must lie in (0, 1), got {self.level}")

    def decompose(self, repeats: Sequence[PairedUnitErrors]) -> UncertaintyDecomposition:
        """Both intervals around the reduction over ``repeats``, and the share the repeats add.

        Each element is one repeat's pair over the same units in the same order; a single
        repeat has nothing to resample at the outer level and adds no share.

        Raises:
            InvalidTwoLevelBootstrapError: If there is no repeat.
            InvalidPairedUnitErrorsError: If the repeats disagree about the units.
        """
        if not repeats:
            raise InvalidTwoLevelBootstrapError("a two-level bootstrap needs a repeat to resample")
        pooled = PairedUnitErrors.pooled(
            [repeat.control for repeat in repeats], [repeat.candidate for repeat in repeats]
        )
        over_units = PairedUnitBootstrap(resamples=self.resamples, seed=self.seed, level=self.level)
        units_only = over_units.reductions(pooled)
        both = sorted(self._over_repeats_and_units(repeats))
        spread_units, spread_both = pvariance(units_only), pvariance(both)
        share = 0.0 if spread_both == 0.0 else 1.0 - spread_units / spread_both
        return UncertaintyDecomposition(
            over_units=over_units.interval_of(units_only),
            over_units_and_repeats=over_units.interval_of(both),
            share_from_repeats=min(1.0, max(0.0, share)),
        )

    def _over_repeats_and_units(self, repeats: Sequence[PairedUnitErrors]) -> list[float]:
        draws = random.Random(self.seed)
        count_repeats, count_units = len(repeats), len(repeats[0].control)
        control_squared = [[e.squared_error for e in repeat.control] for repeat in repeats]
        control_windows = [[e.windows for e in repeat.control] for repeat in repeats]
        candidate_squared = [[e.squared_error for e in repeat.candidate] for repeat in repeats]
        candidate_windows = [[e.windows for e in repeat.candidate] for repeat in repeats]
        reductions = []
        for _ in range(self.resamples):
            picked_repeats = draws.choices(range(count_repeats), k=count_repeats)
            picked_units = draws.choices(range(count_units), k=count_units)
            reductions.append(
                _rmse(control_squared, control_windows, picked_repeats, picked_units)
                - _rmse(candidate_squared, candidate_windows, picked_repeats, picked_units)
            )
        return reductions


def _rmse(
    squared: Sequence[Sequence[float]],
    windows: Sequence[Sequence[int]],
    repeats: Sequence[int],
    units: Sequence[int],
) -> float:
    total = sum(squared[repeat][unit] for repeat in repeats for unit in units)
    count = sum(windows[repeat][unit] for repeat in repeats for unit in units)
    return sqrt(total / count)
