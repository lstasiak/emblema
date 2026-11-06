import random
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.evaluation.domain.exceptions import (
    InvalidAnomalyRankingError,
    InvalidWindowRankingError,
)
from emblema.evaluation.domain.scoring.tied_sweep import TiedSweep
from emblema.evaluation.domain.scoring.unit_counts import UnitCounts

type Window = tuple[int, bool, float, bool]
"""A window in order: its unit, whether it is a hit, what it weighs, whether it closes a tie."""

# Whichever way the misses go, the table or the running count: both are exact.
TABLE, RUNNING = 10**9, 0


def plain_average_precision(windows: Sequence[Window], weights: Sequence[int]) -> float:
    """The sweep one window at a time, in the order of additions the sweep must keep."""
    found = alarmed = area = 0.0
    found_in_group = alarmed_in_group = 0.0
    for unit, hit, worth, closes in windows:
        counted = worth * weights[unit]
        if hit:
            found_in_group += counted
        else:
            alarmed_in_group += counted
        if closes:
            found += found_in_group
            alarmed += alarmed_in_group
            if found_in_group:
                area += found_in_group * found / (found + alarmed)
            found_in_group = alarmed_in_group = 0.0
    return area / found


def plain_auroc(windows: Sequence[Window], weights: Sequence[int]) -> float:
    """The sweep one window at a time, in the order of additions the sweep must keep."""
    below = positives = negatives = tied_positive = tied_negative = 0
    area = 0.0
    for unit, hit, _, closes in windows:
        if hit:
            tied_positive += weights[unit]
        else:
            tied_negative += weights[unit]
        if closes:
            area += tied_positive * (below + tied_negative / 2.0)
            below += tied_negative
            positives += tied_positive
            negatives += tied_negative
            tied_positive = tied_negative = 0
    return area / (positives * negatives)


def unweighed(windows: Sequence[Window]) -> list[Window]:
    """The same windows with every hit weighing one, as a ranking read by the ROC area has them."""
    return [(unit, hit, 1.0, closes) for unit, hit, _, closes in windows]


def sweep(windows: Sequence[Window], units: int, largest_table: int) -> TiedSweep:
    return TiedSweep.of(
        unit_count=units,
        sequence=[window[0] for window in windows],
        hit=[window[1] for window in windows],
        worth=[window[2] for window in windows],
        closes_tie=[window[3] for window in windows],
        largest_table=largest_table,
    )


def drawn_windows(seed: int, units: int, count: int, tie_share: float) -> list[Window]:
    """Windows with at least one hit and one miss on every unit, so no draw can leave either."""
    draws = random.Random(seed)
    windows: list[Window] = []
    for unit in range(units):
        windows.append((unit, True, 1.0 / draws.randint(1, 7), draws.random() >= tie_share))
        windows.append((unit, False, 1.0, draws.random() >= tie_share))
    for _ in range(count):
        hit = draws.random() < 0.2
        worth = 1.0 / draws.randint(1, 7) if hit else 1.0
        windows.append((draws.randrange(units), hit, worth, draws.random() >= tie_share))
    draws.shuffle(windows)
    last = windows[-1]
    windows[-1] = (last[0], last[1], last[2], True)
    return windows


def drawn_weights(seed: int, units: int, draws_of: int) -> list[list[int]]:
    draws = random.Random(seed)
    rows = []
    for _ in range(draws_of):
        row = [0] * units
        for pick in draws.choices(range(units), k=units):
            row[pick] += 1
        row[draws.randrange(units)] += 1
        rows.append(row)
    return rows


@given(seed=st.integers(0, 10_000), units=st.integers(1, 12), count=st.integers(0, 60))
@pytest.mark.parametrize("largest_table", [TABLE, RUNNING])
def test_without_ties_the_areas_are_the_plain_sweeps_to_the_bit(
    largest_table: int, seed: int, units: int, count: int
) -> None:
    windows = drawn_windows(seed, units, count, tie_share=0.0)
    weights = drawn_weights(seed + 1, units, 5)
    swept = sweep(windows, units, largest_table)
    swept_evenly = sweep(unweighed(windows), units, largest_table)
    for row in weights:
        counts = UnitCounts.of_weights(row)
        assert swept.average_precisions(counts) == [plain_average_precision(windows, row)]
        assert swept_evenly.aurocs(counts) == [plain_auroc(windows, row)]


@given(seed=st.integers(0, 10_000), units=st.integers(1, 12), count=st.integers(0, 60))
@pytest.mark.parametrize("largest_table", [TABLE, RUNNING])
def test_under_ties_the_areas_are_the_plain_sweeps(
    largest_table: int, seed: int, units: int, count: int
) -> None:
    windows = drawn_windows(seed, units, count, tie_share=0.5)
    swept = sweep(windows, units, largest_table)
    swept_evenly = sweep(unweighed(windows), units, largest_table)
    for row in drawn_weights(seed + 1, units, 5):
        counts = UnitCounts.of_weights(row)
        assert swept.average_precisions(counts)[0] == pytest.approx(
            plain_average_precision(windows, row), abs=1e-12
        )
        # Whole numbers throughout, so a tie changes nothing about the bits.
        assert swept_evenly.aurocs(counts) == [plain_auroc(windows, row)]


def test_several_draws_read_at_once_answer_as_each_read_alone() -> None:
    windows = drawn_windows(1, units=9, count=200, tie_share=0.1)
    swept = sweep(windows, 9, TABLE)
    picks = [[random.Random(s).randrange(9) for _ in range(9)] + list(range(9)) for s in range(8)]

    together = swept.average_precisions(UnitCounts.of_picks(picks, 9))

    assert together == [swept.average_precisions(UnitCounts.of_picks([one], 9))[0] for one in picks]


def test_a_table_too_large_is_left_for_the_running_count() -> None:
    windows = drawn_windows(2, units=9, count=50, tie_share=0.0)
    row = drawn_weights(3, 9, 1)[0]
    counts = UnitCounts.of_weights(row)
    by_table = sweep(windows, 9, TABLE)
    by_running = sweep(windows, 9, RUNNING)

    assert by_table.aurocs(counts) == by_running.aurocs(counts)
    assert "_table_through" in by_table.__dict__
    assert "_table_through" not in by_running.__dict__


@pytest.mark.parametrize("largest_table", [TABLE, RUNNING])
def test_a_draw_that_leaves_an_outcome_out_is_refused(largest_table: int) -> None:
    windows: list[Window] = [(0, True, 1.0, True), (1, False, 1.0, True)]
    swept = sweep(windows, 2, largest_table)

    for weights in ([0, 1], [1, 0]):
        counts = UnitCounts.of_weights(weights)
        with pytest.raises(InvalidAnomalyRankingError, match="no event or no normal window"):
            swept.average_precisions(counts)
        with pytest.raises(InvalidWindowRankingError, match="one outcome only"):
            swept.aurocs(counts)


def test_picks_count_a_unit_as_often_as_it_is_picked_and_none_at_all_when_absent() -> None:
    counts = UnitCounts.of_picks([[0, 0, 2], []], units=3)

    assert counts.rows.tolist() == [[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


def test_events_found_are_refused_a_draw_without_an_outcome_or_misnamed_hits() -> None:
    windows: list[Window] = [(0, True, 1.0, True), (1, False, 1.0, True)]
    swept = sweep(windows, 2, TABLE)

    with pytest.raises(InvalidAnomalyRankingError, match="2 events were named for 1 hits"):
        swept.event_recalls(UnitCounts.of_weights([1, 1]), share=0.0, hit_events=[1, 2])
    with pytest.raises(InvalidAnomalyRankingError, match="no event or no normal window"):
        swept.event_recalls(UnitCounts.of_weights([1, 0]), share=0.0, hit_events=[1])
    with pytest.raises(InvalidAnomalyRankingError, match="no event or no normal window"):
        swept.event_recalls(UnitCounts.of_weights([0, 1]), share=0.0, hit_events=[1])
    assert swept.event_recalls(UnitCounts.of_weights([1, 1]), share=0.0, hit_events=[1]) == [1.0]
