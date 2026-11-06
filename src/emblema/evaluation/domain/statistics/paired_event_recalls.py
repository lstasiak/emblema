from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.unit_counts import UnitCounts
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings


@dataclass(frozen=True, kw_only=True)
class PairedEventRecalls:
    """Two detectors' rankings of the same units, read by the events each finds at one alarm rate.

    The error of a side is one minus the share of events it finds with at most ``share`` of the
    normal windows flagged, so a reduction is how many more of the events, as a share, an
    operator tolerating that many false alarms is given. The rankings, their repeats and the
    strata a resample draws in are the area's (``PairedAnomalyRankings``), so the same draws
    resample both readings.

    Invariants: the rankings pair as the area's do; the share lies in ``[0, 1]``.

    Attributes:
        rankings: The two sides' rankings, one per repeat.
        share: The share of normal windows the alarm may flag.
    """

    rankings: PairedAnomalyRankings
    share: float

    def __post_init__(self) -> None:
        if not (isfinite(self.share) and 0.0 <= self.share <= 1.0):
            raise InvalidPairedAnomalyRankingsError(
                f"a share of false alarms lies in [0, 1], got {self.share}"
            )

    @property
    def units(self) -> tuple[UnitKey, ...]:
        return self.rankings.units

    @property
    def strata(self) -> tuple[tuple[int, ...], ...]:
        return self.rankings.strata

    @property
    def error_control(self) -> float:
        return self._shortfalls(self.rankings.control, self._once)[0]

    @property
    def error_candidate(self) -> float:
        return self._shortfalls(self.rankings.candidate, self._once)[0]

    @property
    def reduction(self) -> float:
        """How large a share of the events the candidate finds beyond the control."""
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
        return self.reductions_over([picks])[0]

    def reductions_over(self, draws: Sequence[Sequence[int]]) -> list[float]:
        """The reduction over each draw of units, the draws read together."""
        counts = UnitCounts.of_picks(draws, len(self.units))
        return [
            control - candidate
            for control, candidate in zip(
                self._shortfalls(self.rankings.control, counts),
                self._shortfalls(self.rankings.candidate, counts),
                strict=True,
            )
        ]

    def per_repeat_control(self) -> tuple[float, ...]:
        """The shortfall of each repeat of the control on its own."""
        return tuple(
            1.0 - ranking.event_recall_at_false_alarm_share(self.share)
            for ranking in self.rankings.control
        )

    def per_repeat_candidate(self) -> tuple[float, ...]:
        """The shortfall of each repeat of the candidate on its own."""
        return tuple(
            1.0 - ranking.event_recall_at_false_alarm_share(self.share)
            for ranking in self.rankings.candidate
        )

    @property
    def _once(self) -> UnitCounts:
        return UnitCounts.of_weights([1] * len(self.units))

    def _shortfalls(self, repeats: Sequence[AnomalyRanking], counts: UnitCounts) -> list[float]:
        recalls = [ranking.event_recalls(counts, self.share) for ranking in repeats]
        return [1.0 - fmean(over_repeats) for over_repeats in zip(*recalls, strict=True)]
