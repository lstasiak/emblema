import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import sqrt
from statistics import NormalDist
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidKnownRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings


@dataclass(frozen=True, kw_only=True)
class KnownRanking:
    """Paired rankings with a known answer, for checking the procedure rather than a dataset.

    Draws what a cell of a binary task leaves behind, an answer per unit for a control and a
    candidate over one or more repeats, from areas under the ROC curve stated up front. The
    answers are binormal: within each outcome they are normal with unit variance, the positives
    shifted up by ``sqrt(2)`` times the normal quantile of the stated area, the shift that gives
    exactly that area. The variance within an outcome has two components, as in the regression
    generator: a unit effect shared by both sides and every repeat, which is what makes the
    comparison paired, and a residual drawn anew per side and per repeat. Both are inside the
    unit variance, so the stated areas hold whatever share the unit effect takes.

    A third component belongs to the repeat as a whole: a run under another seed is another
    model, and one seed's model can rank every unit better than another's. The resampling of
    units does not see it, so it is kept apart and off by default, to measure what it does to
    an interval stated over units. Each repeat of each side then ranks at its own area, drawn on
    the scale of the normal quantile so that no area leaves the unit interval, and widened there
    by exactly as much as keeps the stated area the expectation over repeats.

    A model can also answer in few distinct values — a small ensemble of trees does — and a tie
    counts half a pair. Asked for, each side's answers are cut into equally likely levels at the
    quantiles of that side's own mixture of the two outcomes, fixed by the population and not by
    the sample, so the area such answers can reach is known exactly and the true reduction is
    the difference of those areas rather than of the stated ones. Levels leave the stated areas
    as the areas before the cut, and are not drawn together with a spread of repeats, under which
    the cut would no longer be fixed.

    Invariants: two or more levels, or none; levels without a spread of repeats.

    Attributes:
        control_auroc: The control's area.
        candidate_auroc: The candidate's area; the true reduction of the error is the
            difference of the two.
        units: How many units each side is scored on, one window each.
        positives: How many of them hold the positive outcome.
        shared: The share of the answers' variance within an outcome that belongs to the unit
            and not to the repeat.
        repeat_spread: About how far one repeat's area strays from its side's stated area, as
            a standard deviation; drawn anew per side and per repeat.
        answer_levels: How many distinct answers each side gives; zero for answers that never
            tie.
    """

    control_auroc: float
    candidate_auroc: float
    units: int = 400
    positives: int = 56
    shared: float = 0.5
    repeat_spread: float = 0.0
    answer_levels: int = 0

    _STANDARD: ClassVar[NormalDist] = NormalDist()

    def __post_init__(self) -> None:
        if self.answer_levels == 1 or self.answer_levels < 0:
            raise InvalidKnownRankingError(
                f"answers need two levels or more to rank, got {self.answer_levels}"
            )
        if self.answer_levels and self.repeat_spread:
            raise InvalidKnownRankingError(
                "levels fixed by the population cannot follow a repeat's own area"
            )

    @property
    def true_reduction(self) -> float:
        return self.reachable_area(self.candidate_auroc) - self.reachable_area(self.control_auroc)

    def reachable_area(self, auroc: float) -> float:
        """The area a side stated at ``auroc`` reaches once its answers are cut into levels."""
        if not self.answer_levels:
            return auroc
        shift = self._shift(auroc)
        edges = (-float("inf"), *self._cuts(shift), float("inf"))
        positive = [
            self._STANDARD.cdf(high - shift) - self._STANDARD.cdf(low - shift)
            for low, high in pairwise(edges)
        ]
        negative = [
            self._STANDARD.cdf(high) - self._STANDARD.cdf(low) for low, high in pairwise(edges)
        ]
        below = area = 0.0
        for chance_positive, chance_negative in zip(positive, negative, strict=True):
            area += chance_positive * (below + chance_negative / 2.0)
            below += chance_negative
        return area

    def paired(self, *, seed: int) -> PairedUnitRankings:
        """One repeat of each side over the units, under ``seed``."""
        return self.repeats(1, seed=seed)

    def repeats(self, count: int, *, seed: int) -> PairedUnitRankings:
        """``count`` repeats of each side over the same units, each with its own luck."""
        draws = random.Random(seed)
        effects = [draws.gauss(0.0, 1.0) for _ in range(self.units)]
        return PairedUnitRankings(
            control=tuple(self._side(self.control_auroc, effects, draws) for _ in range(count)),
            candidate=tuple(self._side(self.candidate_auroc, effects, draws) for _ in range(count)),
        )

    def _side(self, auroc: float, effects: list[float], draws: random.Random) -> WindowRanking:
        # Drawn only when asked for, so a generator without it draws what it always drew.
        if self.repeat_spread:
            auroc = self._repeat_area(auroc, draws)
        shift = self._shift(auroc)
        own = sqrt(1.0 - self.shared)
        common = sqrt(self.shared)
        cuts = self._cuts(shift) if self.answer_levels else ()
        answers = [
            (shift if index < self.positives else 0.0)
            + common * effect
            + own * draws.gauss(0.0, 1.0)
            for index, effect in enumerate(effects)
        ]
        return WindowRanking.of(
            WindowPrediction(
                window=TaskWindow(unit=UnitKey(f"unit/{index:04d}"), position=0, ends_at=0.0),
                target=1.0 if index < self.positives else 0.0,
                predicted=float(bisect_right(cuts, answer)) if cuts else answer,
            )
            for index, answer in enumerate(answers)
        )

    def _shift(self, auroc: float) -> float:
        return sqrt(2.0) * self._STANDARD.inv_cdf(auroc)

    def _cuts(self, shift: float) -> Sequence[float]:
        """The answers that split a side's mixture of outcomes into equally likely levels."""
        return tuple(
            self._mixture_quantile(shift, level / self.answer_levels)
            for level in range(1, self.answer_levels)
        )

    def _mixture_quantile(self, shift: float, share: float) -> float:
        prevalence = self.positives / self.units
        low, high = -12.0, shift + 12.0
        for _ in range(100):
            middle = (low + high) / 2.0
            below = prevalence * self._STANDARD.cdf(middle - shift) + (
                1.0 - prevalence
            ) * self._STANDARD.cdf(middle)
            low, high = (middle, high) if below < share else (low, middle)
        return (low + high) / 2.0

    def _repeat_area(self, auroc: float, draws: random.Random) -> float:
        # A normal quantile scattered by s averages to the area of the quantile shrunk by
        # sqrt(1 + s^2), so the quantile is widened by that factor first; s is the spread asked
        # for in area, carried to the quantile scale by the density at the stated area.
        quantile = self._STANDARD.inv_cdf(auroc)
        scatter = self.repeat_spread / self._STANDARD.pdf(quantile)
        return self._STANDARD.cdf(quantile * sqrt(1.0 + scatter**2) + draws.gauss(0.0, scatter))
