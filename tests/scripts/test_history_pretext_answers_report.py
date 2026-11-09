"""The forecasting pretext's answers against two with no model, scored on hand-laid cells."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.history_pretext_answers_report import (
    ANSWERS,
    ScoredCell,
    compared,
    comparisons_of,
    huber,
    lags,
    same_phase,
    scored,
    table,
)

MEAN = BinnedTokens.FEATURE_NAMES.index("mean")
LAST = BinnedTokens.FEATURE_NAMES.index("last")


def cell(value: float, last: float | None = None) -> np.ndarray:
    """A cell that held readings: present, its five values ``value`` but its last ``last``."""
    features = np.zeros(BinnedTokens.N_FEATURES)
    features[BinnedTokens.PRESENCE] = 1.0
    features[BinnedTokens.VALUE_FEATURES] = value
    features[LAST] = value if last is None else last
    return features


def read_of(bins: list[list[np.ndarray]], answers: np.ndarray, hide: str = "last") -> Any:
    """Cells laid bin by bin over the same channels, as the reading hands them."""
    rows = len(bins[0])
    full = np.array([c for one_bin in bins for c in one_bin])
    hidden = np.zeros((len(bins), rows), dtype=bool)
    if hide == "last":
        hidden[-1] = True
    else:
        hidden[0] = True
    return SimpleNamespace(
        full=full,
        hidden=hidden.reshape(-1),
        channel_ids=np.tile(np.arange(1, rows + 1), len(bins)),
        timeless=np.zeros(len(full), dtype=bool),
        answers=answers,
    )


def test_the_answers_are_scored_on_the_hidden_hour_of_each_channel() -> None:
    nothing = np.zeros(BinnedTokens.N_FEATURES)
    # Channel 1 reads 1, then 3 with a last of 4, then the hidden hour at 5; channel 2 read
    # nothing before its hidden hour, so no answer is scored for it.
    bins = [[cell(1.0), nothing], [cell(3.0, last=4.0), nothing], [cell(5.0), cell(2.0)]]
    answers = np.zeros((6, 5))
    answers[4] = 4.5

    cells, unscored = scored(read_of(bins, answers), bins=3, window=7, axis="M1")

    assert unscored == 1
    (only,) = cells
    assert (only.window, only.axis, only.channel) == (7, "M1", 1)
    assert only.losses[0] == pytest.approx(0.125)  # decoder off by 0.5 on every value
    assert only.losses[1] == pytest.approx(0.5)  # the last reading, 4, off by 1
    assert only.losses[2] == pytest.approx(2.5)  # the median of the means 1 and 3, off by 3
    # Two hours before are too few to find a period: the latest hour, 3 but its last 4, answers.
    assert only.losses[3] == pytest.approx((4 * 1.5 + 0.5) / 5)
    assert only.lag is None


def test_a_draw_that_hid_other_than_the_last_hour_is_refused() -> None:
    bins = [[cell(1.0)], [cell(2.0)]]

    with pytest.raises(ValueError, match="last bin"):
        scored(read_of(bins, np.zeros((2, 5)), hide="first"), bins=2, window=0, axis="M1")


def test_cells_not_laid_bin_by_bin_are_refused() -> None:
    read = read_of([[cell(1.0), cell(1.0)], [cell(2.0), cell(2.0)]], np.zeros((4, 5)))
    read.channel_ids = np.array([1, 2, 2, 1])

    with pytest.raises(ValueError, match="bin by bin"):
        scored(read, bins=2, window=0, axis="M1")


def test_the_huber_loss_is_the_objectives_knee_at_one() -> None:
    assert huber(np.array([0.5, -2.0])) == pytest.approx([0.125, 1.5])


def test_windows_are_resampled_whole_and_the_differences_are_paired() -> None:
    # Two windows: the decoder beats the median by exactly 0.1 on every cell.
    cells = [
        ScoredCell(window=w, axis="M1", channel=c, losses=(0.1 * w, 0.5, 0.1 * w + 0.1, 0.5))
        for w in (1, 2)
        for c in (1, 2, 3)
    ]

    result = compared(cells, "all", np.random.default_rng(0), resamples=200)

    assert (result.windows, result.cells) == (2, 6)
    assert result.means == pytest.approx((0.15, 0.5, 0.25, 0.5))
    difference, low, high = result.differences[1]
    assert (difference, low, high) == pytest.approx((-0.1, -0.1, -0.1))


def test_each_axis_is_compared_on_its_own_beside_all_of_them() -> None:
    cells = [
        ScoredCell(window=0, axis="M1", channel=1, losses=(0.1, 0.2, 0.3, 0.4), lag=3),
        ScoredCell(window=1, axis="M2", channel=9, losses=(0.4, 0.2, 0.3, 0.4)),
    ]

    labels = [c.label for c in comparisons_of(cells, np.random.default_rng(0))]

    assert labels == ["all", "M1", "M2"]
    assert len(ANSWERS) == 4
    assert "| M2 | 1 | 1 |" in table(comparisons_of(cells, np.random.default_rng(0)))
    assert lags(cells).splitlines() == [
        "M1: 0 of 1 cells fell back; periods chosen most: 3 h 100%",
        "M2: 1 of 1 cells fell back; periods chosen most: none",
    ]


def test_no_cell_to_compare_is_refused() -> None:
    with pytest.raises(ValueError, match="no cell"):
        compared([], "M1", np.random.default_rng(0))


def hours_of(series: list[float]) -> np.ndarray:
    return np.repeat(np.array(series, dtype=np.float64)[:, None], 5, axis=1)


def test_a_channel_that_cycles_is_answered_by_its_last_cycle() -> None:
    # A cycle of four hours, 0 1 2 3, over 20 hours: the hidden hour is a 0, as four hours back.
    values = hours_of([0.0, 1.0, 2.0, 3.0] * 5)

    answer, lag = same_phase(values, np.ones(20, dtype=bool))

    assert lag == 4
    assert answer == pytest.approx(np.zeros(5))


def test_a_daily_cycle_is_found_in_the_47_hours_before_the_hidden_one() -> None:
    day = np.random.default_rng(0).normal(size=24).tolist()
    values = hours_of((day * 2)[:47])

    answer, lag = same_phase(values, np.ones(47, dtype=bool))

    assert lag == 24
    assert answer == pytest.approx(np.full(5, day[23]))


def test_the_lag_whose_hour_back_did_not_read_is_passed_over() -> None:
    values = hours_of([0.0, 1.0, 2.0, 3.0] * 5)
    present = np.ones(20, dtype=bool)
    present[16] = False  # four hours before the hidden one

    _, lag = same_phase(values, present)

    assert lag not in (None, 4)


def test_too_few_hours_read_together_fall_back_to_the_latest() -> None:
    present = np.zeros(20, dtype=bool)
    present[[2, 9]] = True
    values = hours_of([float(h) for h in range(20)])

    answer, lag = same_phase(values, present)

    assert lag is None
    assert answer == pytest.approx(np.full(5, 9.0))


def test_a_channel_that_never_read_has_no_same_phase_answer() -> None:
    with pytest.raises(ValueError, match="no hour"):
        same_phase(hours_of([0.0] * 4), np.zeros(4, dtype=bool))
