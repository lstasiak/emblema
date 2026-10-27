from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking


@dataclass(frozen=True, kw_only=True)
class PairedAnomalyRankings:
    """Two detectors' rankings of the same units, so the difference of their areas is paired.

    The error of a side is one minus its area under the precision-recall curve, events weighed
    once each, so a reduction is exactly how much area the detector gains over the control.
    Repeats of a cell are pooled by the mean of their areas, each repeat being its own fitted
    detector. The unit is what a resample draws and carries all its windows; the units are
    split into those holding an event's window and those holding none, and a resample draws
    each stratum on its own, so every resample holds events and normal windows in the
    proportion the side has.

    Invariants: at least one repeat on each side; every ranking of either side names the same
    units and the same windows with the same events.

    Attributes:
        control: The rankings of what the detector is compared against, one per repeat.
        candidate: The rankings of what is compared, one per repeat.
    """

    control: tuple[AnomalyRanking, ...]
    candidate: tuple[AnomalyRanking, ...]

    def __post_init__(self) -> None:
        if not self.control or not self.candidate:
            raise InvalidPairedAnomalyRankingsError(
                "a paired comparison needs a repeat on each side"
            )
        first = self.control[0]
        for ranking in (*self.control, *self.candidate):
            if ranking.units != first.units:
                raise InvalidPairedAnomalyRankingsError("the rankings disagree about the units")
            if ranking.outcomes != first.outcomes:
                raise InvalidPairedAnomalyRankingsError(
                    "the rankings disagree about the windows or their events"
                )

    @property
    def units(self) -> tuple[UnitKey, ...]:
        return self.control[0].units

    @property
    def strata(self) -> tuple[tuple[int, ...], ...]:
        """Indices of the units holding an event's window, then of those holding none."""
        first = self.control[0]
        holding = {unit for unit, event in zip(first.sequence, first.events, strict=True) if event}
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
        return tuple(1.0 - ranking.average_precision for ranking in self.control)

    def per_repeat_candidate(self) -> tuple[float, ...]:
        """The shortfall of each repeat of the candidate on its own."""
        return tuple(1.0 - ranking.average_precision for ranking in self.candidate)

    @staticmethod
    def _shortfall(repeats: Sequence[AnomalyRanking], weights: Sequence[int]) -> float:
        return 1.0 - fmean(ranking.average_precision_weighted(weights) for ranking in repeats)
