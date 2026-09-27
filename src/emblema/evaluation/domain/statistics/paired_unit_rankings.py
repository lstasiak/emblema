from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from emblema.evaluation.domain.exceptions import InvalidPairedUnitRankingsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking


@dataclass(frozen=True, kw_only=True)
class PairedUnitRankings:
    """Two candidates' rankings of the same units, so the difference of their areas is paired.

    The error of a side is one minus its area under the ROC curve, the share of (positive,
    negative) pairs its answers order the wrong way, so a reduction is exactly how much area the
    candidate gains over the control. Repeats of a cell are pooled by the mean of their areas:
    each repeat is its own fitted model, and ranking the answers of five models together would
    compare one model's positives with another's negatives on scales nobody fitted to agree.

    The unit is what a resample draws, as it is for a squared error, and each unit carries all
    its windows. The units are split into two strata — those holding a positive and those that
    do not — and a resample draws each stratum on its own, so every resample holds both outcomes
    in the proportion the side has and no resample is left without an area to compute.

    Invariants: at least one repeat on each side; every ranking of either side names the same
    units and the same windows with the same outcomes, since a pair over different questions is
    not a pair.

    Attributes:
        control: The rankings of what the candidate is compared against, one per repeat.
        candidate: The rankings of what is compared, one per repeat.
    """

    control: tuple[WindowRanking, ...]
    candidate: tuple[WindowRanking, ...]

    def __post_init__(self) -> None:
        if not self.control or not self.candidate:
            raise InvalidPairedUnitRankingsError("a paired comparison needs a repeat on each side")
        first = self.control[0]
        for ranking in (*self.control, *self.candidate):
            if ranking.units != first.units:
                raise InvalidPairedUnitRankingsError("the rankings disagree about the units")
            if ranking.outcomes != first.outcomes:
                raise InvalidPairedUnitRankingsError(
                    "the rankings disagree about the windows or their outcomes"
                )

    @property
    def units(self) -> tuple[UnitKey, ...]:
        return self.control[0].units

    @property
    def strata(self) -> tuple[tuple[int, ...], ...]:
        """Indices of the units holding a positive, then of those that do not."""
        first = self.control[0]
        holding = {
            unit for unit, positive in zip(first.sequence, first.positive, strict=True) if positive
        }
        everyone = range(len(first.units))
        return (
            tuple(unit for unit in everyone if unit in holding),
            tuple(unit for unit in everyone if unit not in holding),
        )

    @property
    def error_control(self) -> float:
        return self._shortfall(self.control, [1] * len(self.units))

    @property
    def error_candidate(self) -> float:
        return self._shortfall(self.candidate, [1] * len(self.units))

    @property
    def reduction(self) -> float:
        """How much area the candidate gains over the control: its error's reduction."""
        return self.error_control - self.error_candidate

    @property
    def relative_reduction(self) -> float | None:
        """The reduction as a share of the control's shortfall; ``None`` where there is none."""
        control = self.error_control
        if control == 0.0:
            return None
        return self.reduction / control

    def reduction_over(self, picks: Sequence[int]) -> float:
        """The reduction over the units at ``picks``, a unit counted as often as it is picked."""
        weights = [0] * len(self.units)
        for pick in picks:
            weights[pick] += 1
        return self._shortfall(self.control, weights) - self._shortfall(self.candidate, weights)

    def per_repeat_control(self) -> tuple[float, ...]:
        """The shortfall of each repeat of the control on its own."""
        return tuple(1.0 - ranking.auroc for ranking in self.control)

    def per_repeat_candidate(self) -> tuple[float, ...]:
        """The shortfall of each repeat of the candidate on its own."""
        return tuple(1.0 - ranking.auroc for ranking in self.candidate)

    @staticmethod
    def _shortfall(repeats: Sequence[WindowRanking], weights: Sequence[int]) -> float:
        return 1.0 - fmean(ranking.auroc_weighted(weights) for ranking in repeats)
