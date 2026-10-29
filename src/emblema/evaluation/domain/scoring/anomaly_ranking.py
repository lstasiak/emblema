from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidAnomalyRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class AnomalyRanking:
    """One detection run's scores put in order, to read how well they set the events apart.

    The windows are swept from the highest score down, as alarms would be raised by lowering a
    threshold. Every event weighs one, spread evenly over the windows it spans, and every normal
    window weighs one, so an event of a month and an event of an hour count the same and a
    false alarm costs the same at any hour. The area under the precision-recall curve is the
    average over the events, each weighed once, of the precision at the threshold that reaches
    it: a tie between windows is swept as one group, every window of the group getting the
    group's precision. A resample of units changes only how many times each unit counts, so the
    sweep takes a weight per unit and the sorting is never repeated.

    Two more readings come off the same order. The share of events found when the alarm is set
    so that a stated share of the normal windows is flagged says what the detector gives an
    operator who will tolerate that many false alarms. The benchmark's own score reads the run
    at the alarm it set: a window with a score above zero is an alarm, an event is found when
    one of its windows is, a run of consecutive flagged windows of one unit that touches no
    event is a false alarm, and the precision over events is corrected by the share of normal
    windows left quiet, so that flagging everything scores nothing.

    Units are held in their own order, the order a paired comparison names them in, so two
    rankings of the same units weigh the same unit by the same index.

    Invariants: at least one window; every target is an event's number or zero; the windows
    hold an event and a normal window, since an area needs both.

    Attributes:
        units: Every unit the windows belong to, in unit order.
        sequence: Index into ``units`` of each window, windows in descending order of score.
        events: The event each window of ``sequence`` overlaps, zero for none.
        weights: What each window of ``sequence`` weighs: one over the event's windows, or one.
        alarms: Whether each window of ``sequence`` scored above zero.
        positions: Where the block holds each window of ``sequence``, so runs of alarms can be
            told from scattered ones.
        closes_tie: Whether each window of ``sequence`` is the last of its group of equal scores.
        outcomes: Each window's event in window order (unit, then position), so two rankings
            can be checked to answer the same question.
    """

    units: tuple[UnitKey, ...]
    sequence: tuple[int, ...]
    events: tuple[int, ...]
    weights: tuple[float, ...]
    alarms: tuple[bool, ...]
    positions: tuple[int, ...]
    closes_tie: tuple[bool, ...]
    outcomes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sequence:
            raise InvalidAnomalyRankingError("a ranking needs a window")
        lengths = {
            len(column)
            for column in (
                self.sequence,
                self.events,
                self.weights,
                self.alarms,
                self.positions,
                self.closes_tie,
                self.outcomes,
            )
        }
        if len(lengths) != 1:
            raise InvalidAnomalyRankingError("every window needs an event, a weight and a place")
        if not any(self.events) or all(self.events):
            raise InvalidAnomalyRankingError("a ranking needs an event and a normal window")

    @classmethod
    def of(cls, predictions: Iterable[WindowPrediction]) -> Self:
        """The ranking of ``predictions`` by their scores.

        Raises:
            InvalidAnomalyRankingError: If there is no prediction, a target is not an event's
                number, or the predictions hold no event or no normal window.
        """
        rows = sorted(predictions, key=lambda p: (str(p.window.unit), p.window.position))
        for row in rows:
            if row.target != int(row.target) or row.target < AnomalyScheme.NORMAL:
                raise InvalidAnomalyRankingError(
                    f"a window of {row.window.unit} has target {row.target}, not an event number"
                )
        spans: dict[int, int] = {}
        for row in rows:
            if row.target != AnomalyScheme.NORMAL:
                spans[int(row.target)] = spans.get(int(row.target), 0) + 1
        units = tuple(sorted({row.window.unit for row in rows}, key=str))
        index = {unit: place for place, unit in enumerate(units)}
        ranked = sorted(rows, key=lambda row: -row.predicted)
        return cls(
            units=units,
            sequence=tuple(index[row.window.unit] for row in ranked),
            events=tuple(int(row.target) for row in ranked),
            weights=tuple(
                1.0 / spans[int(row.target)] if row.target != AnomalyScheme.NORMAL else 1.0
                for row in ranked
            ),
            alarms=tuple(row.predicted > 0.0 for row in ranked),
            positions=tuple(row.window.position for row in ranked),
            closes_tie=tuple(
                place == len(ranked) - 1 or ranked[place + 1].predicted != row.predicted
                for place, row in enumerate(ranked)
            ),
            outcomes=tuple(int(row.target) for row in rows),
        )

    @property
    def event_count(self) -> int:
        return len(set(self.events) - {0})

    @property
    def average_precision(self) -> float:
        """The area under the precision-recall curve, each unit counted once."""
        return self.average_precision_weighted([1] * len(self.units))

    def average_precision_weighted(self, weights: Sequence[int]) -> float:
        """The area with the windows of unit ``i`` counted ``weights[i]`` times over.

        Raises:
            InvalidAnomalyRankingError: If the weights leave no event or no normal window.
        """
        found = alarmed = 0.0
        area = 0.0
        found_in_group = alarmed_in_group = 0.0
        for unit, event, weight, closes in zip(
            self.sequence, self.events, self.weights, self.closes_tie, strict=True
        ):
            counted = weight * weights[unit]
            if event:
                found_in_group += counted
            else:
                alarmed_in_group += counted
            if closes:
                found += found_in_group
                alarmed += alarmed_in_group
                if found_in_group:
                    area += found_in_group * found / (found + alarmed)
                found_in_group = alarmed_in_group = 0.0
        if found == 0.0 or alarmed == 0.0:
            raise InvalidAnomalyRankingError("the weights leave no event or no normal window")
        return area / found

    def event_recall_at_false_alarm_share(self, share: float) -> float:
        """The share of events found with at most ``share`` of the normal windows flagged.

        The alarm is lowered one group of equal scores at a time; it stops before the group that
        would take the flagged share of normal windows past ``share``.

        Raises:
            InvalidAnomalyRankingError: If ``share`` is not in ``[0, 1]``.
        """
        if not 0.0 <= share <= 1.0:
            raise InvalidAnomalyRankingError(f"a share of false alarms lies in [0, 1], got {share}")
        normal = sum(1 for event in self.events if not event)
        allowed = share * normal
        flagged_normal = 0
        found: set[int] = set()
        found_in_group: set[int] = set()
        normal_in_group = 0
        for event, closes in zip(self.events, self.closes_tie, strict=True):
            if event:
                found_in_group.add(event)
            else:
                normal_in_group += 1
            if closes:
                if flagged_normal + normal_in_group > allowed:
                    break
                flagged_normal += normal_in_group
                found |= found_in_group
                found_in_group = set()
                normal_in_group = 0
        return len(found) / self.event_count

    def corrected_event_fbeta(self, beta: float) -> float:
        """The benchmark's event-wise F-score at the alarm the run set, corrected for quiet time.

        Recall is the share of events with a flagged window. Precision is the share of true
        events among the events found and the false alarms, a false alarm being a run of
        consecutive flagged windows of one unit that overlaps no event, multiplied by the share
        of normal windows left unflagged, so that a run that flags everything scores nothing.

        Raises:
            InvalidAnomalyRankingError: If ``beta`` is not positive.
        """
        if beta <= 0.0:
            raise InvalidAnomalyRankingError(f"beta must be positive, got {beta}")
        found = {
            event for event, alarm in zip(self.events, self.alarms, strict=True) if alarm and event
        }
        recall = len(found) / self.event_count
        normal = [not event for event in self.events]
        quiet = sum(
            1
            for is_normal, alarm in zip(normal, self.alarms, strict=True)
            if is_normal and not alarm
        )
        true_negative_rate = quiet / sum(normal)
        false_alarms = self._false_alarm_runs()
        hit = len(found)
        if hit + false_alarms == 0:
            return 0.0
        precision = hit / (hit + false_alarms) * true_negative_rate
        if precision + recall == 0.0:
            return 0.0
        return (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall)

    def _false_alarm_runs(self) -> int:
        """Runs of consecutive flagged windows of one unit that overlap no event."""
        flagged: dict[int, list[tuple[int, int]]] = {}
        for unit, event, alarm, position in zip(
            self.sequence, self.events, self.alarms, self.positions, strict=True
        ):
            if alarm:
                flagged.setdefault(unit, []).append((position, event))
        runs = 0
        for windows in flagged.values():
            windows.sort()
            previous: int | None = None
            run_holds_event = False
            for position, event in windows:
                if previous is None or position != previous + 1:
                    if previous is not None and not run_holds_event:
                        runs += 1
                    run_holds_event = False
                run_holds_event = run_holds_event or bool(event)
                previous = position
            if previous is not None and not run_holds_event:
                runs += 1
        return runs
