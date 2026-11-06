import random
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.evaluation.domain.exceptions import InvalidAnomalyRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.unit_counts import UnitCounts
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from tests.evaluation.support import prediction

type Row = tuple[str, int, float, float]
"""A window: its unit, its position, the event it overlaps (zero for none) and its score."""


def ranked(*rows: Row) -> AnomalyRanking:
    return AnomalyRanking.of(
        prediction(unit, position, event, score) for unit, position, event, score in rows
    )


def by_definition(rows: Sequence[Row], unit_weights: dict[str, int] | None = None) -> float:
    """The area by its definition: precision at each event's windows, events weighing one each.

    Written as the mean over the events' windows of the precision at the threshold that admits
    the window, each window of an event weighing one over the event's windows; ties are swept
    as a group, so a window's precision is the group's. A unit counted twice is there twice,
    windows and events alike, so an event of a unit counted twice weighs two.
    """
    weights = unit_weights or {}
    spans: dict[float, float] = {}
    for _unit, _, event, _ in rows:
        if event:
            spans[event] = spans.get(event, 0.0) + 1
    weighed = [
        (score, event, weights.get(unit, 1) * (1.0 / spans[event] if event else 1.0))
        for unit, _, event, score in rows
    ]
    area = total = 0.0
    for score, event, weight in weighed:
        if not event:
            continue
        admitted = [
            (other_event, other_weight)
            for other, other_event, other_weight in weighed
            if other >= score
        ]
        hits = sum(w for e, w in admitted if e)
        alarms = sum(w for e, w in admitted if not e)
        area += weight * hits / (hits + alarms)
        total += weight
    return area / total


def test_scores_that_put_every_event_above_every_normal_window_are_a_perfect_area() -> None:
    assert (
        ranked(
            ("a", 0, 0, 0.1), ("a", 1, 1, 0.9), ("b", 0, 0, 0.2), ("b", 1, 2, 0.8)
        ).average_precision
        == 1.0
    )


def test_scores_that_put_the_event_below_every_normal_window_score_its_share() -> None:
    # One event among three windows, admitted last: precision one in three.
    assert ranked(
        ("a", 0, 0, 0.9), ("a", 1, 0, 0.8), ("b", 0, 1, 0.1)
    ).average_precision == pytest.approx(1 / 3)


def test_an_event_weighs_one_however_many_windows_it_spans() -> None:
    # A three-window event found first, a normal window, then a one-window event: the first
    # event's windows weigh a third each, so the area is the mean of 1 and 2/3, not of 1, 1, 1
    # and 2/3.
    rows = (
        ("a", 0, 1, 0.9),
        ("a", 1, 1, 0.8),
        ("a", 2, 1, 0.7),
        ("a", 3, 0, 0.6),
        ("b", 0, 2, 0.5),
        ("b", 1, 0, 0.1),
    )

    assert ranked(*rows).average_precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert ranked(*rows).average_precision == pytest.approx(by_definition(rows))


def test_a_tie_is_swept_as_one_group_with_the_groups_precision() -> None:
    # An event's window tied with a normal one at the top: both admitted together, precision
    # one half for the event.
    rows = (("a", 0, 1, 0.5), ("a", 1, 0, 0.5), ("b", 0, 0, 0.1))

    assert ranked(*rows).average_precision == pytest.approx(0.5)
    assert ranked(*rows).average_precision == pytest.approx(by_definition(rows))


def test_a_unit_counted_twice_weighs_as_if_its_windows_were_there_twice() -> None:
    rows = (("a", 0, 1, 0.9), ("a", 1, 0, 0.6), ("b", 0, 0, 0.7), ("b", 1, 2, 0.3))
    ranking = ranked(*rows)

    twice = ranking.average_precision_weighted([2, 1])

    assert twice == pytest.approx(by_definition(rows, {"a": 2}))
    assert ranking.average_precision_weighted([1, 1]) == pytest.approx(by_definition(rows))


def test_weights_that_leave_no_event_or_no_normal_window_are_refused() -> None:
    ranking = ranked(("a", 0, 1, 0.9), ("b", 0, 0, 0.1))

    with pytest.raises(InvalidAnomalyRankingError, match="leave no event"):
        ranking.average_precision_weighted([0, 1])
    with pytest.raises(InvalidAnomalyRankingError, match="leave no event"):
        ranking.average_precision_weighted([1, 0])


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["a", "b", "c"]),
            st.integers(min_value=0, max_value=3),
            st.integers(min_value=0, max_value=2),
            st.sampled_from([0.1, 0.2, 0.3, 0.5, 0.9]),
        ),
        min_size=2,
        max_size=12,
        unique_by=lambda row: (row[0], row[1]),
    )
)
def test_the_sweep_agrees_with_the_definition(rows: list[tuple[str, int, int, float]]) -> None:
    events = {event for _, _, event, _ in rows}
    if 0 not in events or events == {0}:
        return

    assert ranked(*rows).average_precision == pytest.approx(by_definition(rows))


def test_units_are_named_in_their_own_order_and_the_outcomes_in_window_order() -> None:
    ranking = ranked(("b", 1, 2, 0.9), ("a", 0, 0, 0.1), ("b", 0, 0, 0.5))

    assert ranking.units == (UnitKey("a"), UnitKey("b"))
    assert ranking.outcomes == (0, 0, 2)
    assert ranking.event_count == 1


def test_the_events_found_at_a_share_of_false_alarms_stops_before_the_share_is_passed() -> None:
    # Events first, second and fourth from the top, normal windows between: at no false alarm
    # one event; at half the normal windows two; at every normal window all three.
    ranking = ranked(
        ("a", 0, 1, 0.9),
        ("a", 1, 0, 0.8),
        ("a", 2, 2, 0.7),
        ("a", 3, 0, 0.6),
        ("b", 0, 3, 0.5),
    )

    assert ranking.event_recall_at_false_alarm_share(0.0) == pytest.approx(1 / 3)
    assert ranking.event_recall_at_false_alarm_share(0.5) == pytest.approx(2 / 3)
    assert ranking.event_recall_at_false_alarm_share(1.0) == 1.0


def test_a_share_of_false_alarms_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(InvalidAnomalyRankingError, match="lies in"):
        ranked(("a", 0, 1, 0.9), ("a", 1, 0, 0.1)).event_recall_at_false_alarm_share(1.5)


def test_the_event_wise_score_counts_a_run_of_alarms_once_and_corrects_for_quiet_time() -> None:
    # Unit a: three consecutive alarms over positions 0-2, one of them an event's window, so the
    # run hits and is no false alarm; a quiet normal window at 3. Unit b: one alarm on a normal
    # window, a false alarm; an event missed at 1. Events found 1 of 2; false alarms 1; normal
    # windows 4, of which 1 quiet (a3): precision 1/2 x 1/4 = 1/8; recall 1/2.
    ranking = ranked(
        ("a", 0, 0, 0.5),
        ("a", 1, 1, 0.7),
        ("a", 2, 0, 0.2),
        ("a", 3, 0, -0.1),
        ("b", 0, 0, 0.3),
        ("b", 1, 2, -0.4),
    )

    precision, recall = 1 / 8, 1 / 2
    expected = 1.25 * precision * recall / (0.25 * precision + recall)
    assert ranking.corrected_event_fbeta(0.5) == pytest.approx(expected)


def test_flagging_everything_scores_nothing_and_flagging_nothing_scores_nothing() -> None:
    everything = ranked(("a", 0, 0, 0.5), ("a", 1, 1, 0.7), ("b", 0, 0, 0.3))
    nothing = ranked(("a", 0, 0, -0.5), ("a", 1, 1, -0.7), ("b", 0, 0, -0.3))

    assert everything.corrected_event_fbeta(0.5) == 0.0
    assert nothing.corrected_event_fbeta(0.5) == 0.0


def test_two_separate_runs_of_alarms_on_normal_windows_are_two_false_alarms() -> None:
    # Positions 0 and 2 of unit a alarmed with 1 quiet between them: two runs, neither hitting
    # the event at unit b, so precision is zero and so is the score.
    ranking = ranked(("a", 0, 0, 0.5), ("a", 1, 0, -0.5), ("a", 2, 0, 0.5), ("b", 0, 1, 0.9))

    assert ranking._false_alarm_runs() == 2
    assert ranking.corrected_event_fbeta(0.5) == pytest.approx(
        1.25 * (1 / 3 * 1 / 3) * 1.0 / (0.25 * (1 / 3 * 1 / 3) + 1.0)
    )


def test_a_beta_that_is_not_positive_is_refused() -> None:
    with pytest.raises(InvalidAnomalyRankingError, match="beta"):
        ranked(("a", 0, 1, 0.9), ("a", 1, 0, 0.1)).corrected_event_fbeta(0.0)


def test_a_ranking_needs_an_event_and_a_normal_window() -> None:
    with pytest.raises(InvalidAnomalyRankingError, match="needs a window"):
        AnomalyRanking.of(())
    with pytest.raises(InvalidAnomalyRankingError, match="an event and a normal window"):
        ranked(("a", 0, 1, 0.9), ("a", 1, 2, 0.1))
    with pytest.raises(InvalidAnomalyRankingError, match="an event and a normal window"):
        ranked(("a", 0, 0, 0.9), ("a", 1, 0, 0.1))


def test_a_target_that_is_no_event_number_is_refused() -> None:
    with pytest.raises(InvalidAnomalyRankingError, match="not an event number"):
        AnomalyRanking.of(
            [
                WindowPrediction(
                    window=prediction("a", 0, 0.0, 0.1).window, target=0.5, predicted=0.1
                ),
                prediction("a", 1, 1.0, 0.9),
            ]
        )


def copied(rows: Sequence[Row], counts: Sequence[int], units: Sequence[str]) -> list[Row]:
    """The rows with unit ``units[i]`` there ``counts[i]`` times, each copy's events its own."""
    out: list[Row] = []
    for place, unit in enumerate(units):
        for copy in range(counts[place]):
            out.extend(
                (f"{unit}#{copy}", position, event * 100 + copy if event else 0, score)
                for row_unit, position, event, score in rows
                if row_unit == unit
            )
    return out


def drawn_rows(seed: int, tied: bool) -> list[Row]:
    """Four units, each with a normal window and events of one to three windows, scores tied
    when rounded to a tenth."""
    draws = random.Random(seed)

    def score() -> float:
        return round(draws.random(), 1 if tied else 6)

    rows: list[Row] = []
    event = 0
    for unit in UNITS:
        rows.append((unit, 0, 0, score()))
        for position in range(1, draws.randint(3, 9)):
            if draws.random() < 0.25:
                event += 1
                rows.extend(
                    (unit, position * 10 + length, event, score())
                    for length in range(draws.randint(1, 3))
                )
            else:
                rows.append((unit, position * 10, 0, score()))
    if not event:
        rows.append(("a", 999, 1, 0.5))
    return rows


UNITS = ("a", "b", "c", "d")


@given(
    seed=st.integers(0, 10_000),
    share=st.sampled_from([0.0, 0.01, 0.1, 0.3, 1.0]),
    tied=st.booleans(),
)
def test_the_events_found_under_draws_are_the_plain_sweeps_over_the_drawn_copies(
    seed: int, share: float, tied: bool
) -> None:
    rows = drawn_rows(seed, tied)
    ranking = ranked(*rows)
    draws = random.Random(seed + 1)
    held = []
    for _ in range(6):
        counts = [draws.randint(0, 3) for _ in UNITS]
        events = {event for _, _, event, _ in copied(rows, counts, UNITS)}
        if 0 in events and len(events) > 1:
            held.append(counts)

    expected = [
        ranked(*copied(rows, counts, UNITS)).event_recall_at_false_alarm_share(share)
        for counts in held
    ]

    picks = [[place for place, count in enumerate(row) for _ in range(count)] for row in held]
    found = ranking.event_recalls(UnitCounts.of_picks(picks, len(UNITS)), share) if held else []
    assert found == pytest.approx(expected, abs=1e-12)
    assert ranking.event_recalls(UnitCounts.of_weights([1] * len(UNITS)), share) == [
        ranking.event_recall_at_false_alarm_share(share)
    ]


def test_an_event_over_two_units_is_found_once_in_either() -> None:
    ranking = ranked(("a", 0, 1, 0.9), ("b", 0, 1, 0.1), ("a", 1, 0, 0.5), ("b", 1, 0, 0.4))

    assert ranking.event_recalls(UnitCounts.of_weights([1, 1]), 0.0) == [1.0]
    assert ranking.event_recall_at_false_alarm_share(0.0) == 1.0
    # Without the unit holding the flagged window, the event is not found.
    assert ranking.event_recalls(UnitCounts.of_weights([0, 2]), 0.0) == [0.0]


def test_a_share_of_false_alarms_outside_zero_to_one_is_refused_under_draws() -> None:
    with pytest.raises(InvalidAnomalyRankingError, match=r"\[0, 1\]"):
        ranked(("a", 0, 1, 0.9), ("a", 1, 0, 0.1)).event_recalls(UnitCounts.of_weights([1]), 1.5)
