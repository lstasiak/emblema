"""The forecasting pretext's answers against those with no model and a ridge, on hand-laid cells."""

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.history_pretext_answers_report import (
    COLUMNS,
    ChannelRidge,
    FitRow,
    RidgeFit,
    ScoredCell,
    WindowSet,
    fitting_rows,
    huber,
    lags,
    same_phase,
    scored,
    write_cells,
    write_context,
)

MEAN = BinnedTokens.FEATURE_NAMES.index("mean")
LAST = BinnedTokens.FEATURE_NAMES.index("last")
FEATURES = BinnedTokens.N_FEATURES


def cell(value: float, last: float | None = None) -> np.ndarray:
    """A cell that held readings: present, its five values ``value`` but its last ``last``."""
    features = np.zeros(FEATURES)
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


def constant_ridge(
    hours: int, channels: tuple[int, ...], answer: float, window: bool = False
) -> ChannelRidge:
    """A ridge whose every channel answers ``answer`` whatever its history, clipping nothing."""
    ridge = ChannelRidge(hours, "abc", window)
    for channel in channels:
        companions = np.array([c for c in channels if c != channel], dtype=np.int64)
        features = ridge.features_of(len(companions))
        weights = np.zeros((features, 5))
        weights[0] = answer
        ridge.weights[channel] = weights
        ridge.penalties[channel] = 1.0
        ridge.rows[channel] = 10
        ridge.companions[channel] = companions
        ridge.low[channel] = np.full(features, -np.inf)
        ridge.high[channel] = np.full(features, np.inf)
    return ridge


def asked(channel: int, history: np.ndarray, others: dict[int, np.ndarray] | None = None) -> FitRow:
    """A row to answer: the channel's history and, where given, the other channels' means."""
    companions = np.array(sorted(others or {}), dtype=np.int64)
    means = (
        np.stack([others[c] for c in companions], axis=1) if others else np.zeros((len(history), 0))
    )
    return FitRow(channel, history, means, companions, np.zeros(5), 0)


def test_the_answers_are_scored_on_the_hidden_hour_of_each_channel() -> None:
    nothing = np.zeros(FEATURES)
    # Channel 1 reads 1, then 3 with a last of 4, then the hidden hour at 5; channel 2 read
    # nothing before its hidden hour, so no answer is scored for it.
    bins = [[cell(1.0), nothing], [cell(3.0, last=4.0), nothing], [cell(5.0), cell(2.0)]]
    answers = np.zeros((6, 5))
    answers[4] = 4.5
    ridge = constant_ridge(2, (1, 2), 6.0)
    window_ridge = constant_ridge(2, (1, 2), 7.0, window=True)

    cells, unscored = scored(
        read_of(bins, answers),
        bins=3,
        window=7,
        axis="M1",
        ridge=ridge,
        window_ridge=window_ridge,
    )

    assert unscored == 1
    (only,) = cells
    assert (only.window, only.axis, only.channel, only.target_mean) == (7, "M1", 1, 5.0)
    assert only.losses[0] == pytest.approx(0.125)  # decoder off by 0.5 on every value
    assert only.losses[1] == pytest.approx(0.5)  # the last reading, 4, off by 1
    assert only.losses[2] == pytest.approx(2.5)  # the median of the means 1 and 3, off by 3
    # Two hours before are too few to find a period: the latest hour, 3 but its last 4, answers.
    assert only.losses[3] == pytest.approx((4 * 1.5 + 0.5) / 5)
    assert only.losses[4] == pytest.approx(0.5)  # the ridge's 6, off by 1
    assert only.losses[5] == pytest.approx(1.5)  # the window ridge's 7, off by 2
    assert only.means == pytest.approx((4.5, 4.0, 2.0, 3.0, 6.0, 7.0))
    assert only.lag is None
    assert only.history == pytest.approx((1.0, 3.0))


def test_a_draw_that_hid_other_than_the_last_hour_is_refused() -> None:
    bins = [[cell(1.0)], [cell(2.0)]]
    ridge = constant_ridge(1, (1,), 0.0)

    with pytest.raises(ValueError, match="last bin"):
        scored(
            read_of(bins, np.zeros((2, 5)), hide="first"),
            bins=2,
            window=0,
            axis="M1",
            ridge=ridge,
            window_ridge=ridge,
        )


def test_cells_not_laid_bin_by_bin_are_refused() -> None:
    read = read_of([[cell(1.0), cell(1.0)], [cell(2.0), cell(2.0)]], np.zeros((4, 5)))
    read.channel_ids = np.array([1, 2, 2, 1])

    ridge = constant_ridge(1, (1, 2), 0.0)

    with pytest.raises(ValueError, match="bin by bin"):
        scored(read, bins=2, window=0, axis="M1", ridge=ridge, window_ridge=ridge)


def test_the_huber_loss_is_the_objectives_knee_at_one() -> None:
    assert huber(np.array([0.5, -2.0])) == pytest.approx([0.125, 1.5])
    assert huber(np.array([0.5, -2.0]), delta=0.25) == pytest.approx([0.09375, 0.46875])


def test_each_axis_reports_the_periods_its_cells_chose() -> None:
    cells = [
        ScoredCell(0, "M1", 1, 0.0, (0.1,) * 6, (0.0,) * 6, lag=3),
        ScoredCell(1, "M2", 9, 0.0, (0.4,) * 6, (0.0,) * 6),
    ]

    assert lags(cells).splitlines() == [
        "M1: 0 of 1 cells fell back; periods chosen most: 3 h 100%",
        "M2: 1 of 1 cells fell back; periods chosen most: none",
    ]


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


def linear_rows(
    rng: np.random.Generator,
    count: int,
    channel: int,
    slope: float,
    other: tuple[int, float] | None = None,
) -> list[FitRow]:
    """Cells whose five values are ``slope`` times the last hour's mean plus one, nearly.

    With ``other`` (a companion channel and its slope), that channel's last hourly mean joins
    the rule, and the row carries the companion's means.
    """
    rows = []
    for k in range(count):
        history = rng.normal(size=(2, FEATURES))
        value = slope * history[-1, MEAN] + 1.0
        companions = np.zeros(0, dtype=np.int64)
        means = np.zeros((2, 0))
        if other is not None:
            means = rng.normal(size=(2, 1))
            companions = np.array([other[0]], dtype=np.int64)
            value += other[1] * means[-1, 0]
        target = np.full(5, value) + rng.normal(scale=0.01, size=5)
        rows.append(FitRow(channel, history, means, companions, target, k % 2))
    return rows


def fitted(hours: int, corpus: str, rows: list[FitRow], *, window: bool = False) -> ChannelRidge:
    """The ridges fitted on ``rows`` through the same accumulator the report uses."""
    fit = RidgeFit(ChannelRidge(hours, corpus, window))
    for row in rows:
        fit.add(row)
    return fit.fitted()


def test_the_ridge_recovers_a_linear_rule_per_channel() -> None:
    rng = np.random.default_rng(0)
    rows = linear_rows(rng, 400, 1, 2.0) + linear_rows(rng, 400, 2, -3.0)

    ridge = fitted(2, "abc", rows)

    history = np.zeros((2, FEATURES))
    history[-1, MEAN] = 0.5
    assert ridge.answer(asked(1, history)) == pytest.approx(np.full(5, 2.0), abs=0.02)
    assert ridge.answer(asked(2, history)) == pytest.approx(np.full(5, -0.5), abs=0.02)
    assert ridge.penalties[1] == min(ChannelRidge.PENALTIES)
    assert ridge.rows == {1: 400, 2: 400}
    assert "the channel alone, 19 features, 2 channels, 800 cells fitted" in ridge.table()
    assert "channels at the grid's edges: 1, 2" in ridge.table()


def test_an_answer_clips_each_feature_to_the_range_the_fit_saw() -> None:
    rows = linear_rows(np.random.default_rng(0), 400, 1, 2.0)
    ridge = fitted(2, "abc", rows)
    greatest = max(row.history[-1, MEAN] for row in rows)

    at_edge, far_out = (np.zeros((2, FEATURES)) for _ in range(2))
    at_edge[-1, MEAN] = greatest
    far_out[-1, MEAN] = 100.0

    assert ridge.answer(asked(1, far_out)) == pytest.approx(ridge.answer(asked(1, at_edge)))
    assert ridge.answer(asked(1, far_out)) == pytest.approx(np.full(5, 2 * greatest + 1), abs=0.05)
    assert ridge.high[1][1 + FEATURES + MEAN] == pytest.approx(greatest)


def test_the_window_ridge_reads_what_another_channel_tells() -> None:
    rows = linear_rows(np.random.default_rng(0), 400, 1, 0.0, other=(2, 3.0))

    alone = fitted(2, "abc", rows)
    window = fitted(2, "abc", rows, window=True)

    history = np.zeros((2, FEATURES))
    row = asked(1, history, {2: np.array([0.0, 0.5])})
    assert window.answer(row) == pytest.approx(np.full(5, 2.5), abs=0.05)
    assert alone.answer(row) == pytest.approx(np.full(5, 1.0), abs=0.3)
    assert window.features_of(1) == 1 + 2 * FEATURES + 2
    assert "the window's other channels beside, 21 features" in window.table()
    assert window.companions == {1: np.array([2])}
    # A window lacking the companion gives it zeros; one holding a channel never fitted with
    # is refused, at an answer and at the fit alike.
    assert window.answer(asked(1, history)) == pytest.approx(
        window.answer(asked(1, history, {2: np.zeros(2)}))
    )
    with pytest.raises(ValueError, match=r"not fitted with channels \[3\]"):
        window.answer(asked(1, history, {3: np.array([0.0, 0.5])}))
    with pytest.raises(ValueError, match="1 other channels"):
        window.described(FitRow(1, history, np.zeros((2, 2)), np.array([2]), np.zeros(5), 0))
    wider = linear_rows(np.random.default_rng(1), 4, 1, 0.0, other=(3, 1.0))
    with pytest.raises(ValueError, match=r"not fitted with channels \[3\]"):
        fitted(2, "abc", rows + wider, window=True)


def test_rows_flushed_in_batches_fit_the_same_ridge(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = linear_rows(np.random.default_rng(0), 300, 1, 2.0, other=(2, 1.0))
    whole = fitted(2, "abc", rows, window=True)

    monkeypatch.setattr(ChannelRidge, "FLUSH_ENTRIES", 500)
    batched = fitted(2, "abc", rows, window=True)

    assert batched.weights[1] == pytest.approx(whole.weights[1])
    assert batched.penalties == whole.penalties
    assert batched.rows == whole.rows
    assert batched.low[1] == pytest.approx(whole.low[1])
    assert batched.high[1] == pytest.approx(whole.high[1])


def test_a_channel_seen_in_one_half_takes_the_middle_penalty() -> None:
    rows = [
        FitRow(5, row.history, row.means, row.companions, row.target, 0)
        for row in linear_rows(np.random.default_rng(1), 50, 5, 1.0)
    ]

    ridge = fitted(2, "abc", rows)

    assert ridge.penalties[5] == ChannelRidge.PENALTIES[len(ChannelRidge.PENALTIES) // 2] == 10.0


def test_the_ridge_refuses_a_history_of_another_shape_and_a_channel_never_fitted() -> None:
    ridge = constant_ridge(3, (1,), 0.0)

    with pytest.raises(ValueError, match="3 hours"):
        ridge.answer(asked(1, np.zeros((2, FEATURES))))
    with pytest.raises(ValueError, match="channel 7"):
        ridge.answer(asked(7, np.zeros((3, FEATURES))))


def test_the_ridge_saved_is_the_ridge_loaded(tmp_path: Path) -> None:
    rows = linear_rows(np.random.default_rng(0), 100, 3, 1.5, other=(4, 1.0))
    ridge = fitted(2, "abc", rows, window=True)

    ridge.save(tmp_path / "ridge.npz")
    loaded = ChannelRidge.load(tmp_path / "ridge.npz")

    assert (loaded.hours, loaded.corpus, loaded.window) == (2, "abc", True)
    assert loaded.weights[3] == pytest.approx(ridge.weights[3])
    assert loaded.penalties == ridge.penalties
    assert loaded.rows == ridge.rows
    assert loaded.companions[3].tolist() == [4]
    assert loaded.low[3] == pytest.approx(ridge.low[3])
    assert loaded.high[3] == pytest.approx(ridge.high[3])


def test_fitting_rows_are_the_cells_the_scoring_would_score() -> None:
    nothing = np.zeros(FEATURES)
    bins = [[cell(1.0), nothing], [cell(3.0), nothing], [cell(5.0), cell(2.0)]]

    rows = fitting_rows(read_of(bins, np.zeros((6, 5))), bins=3, half=1)

    (row,) = rows
    assert (row.channel, row.half) == (1, 1)
    assert row.history.shape == (2, FEATURES)
    assert row.history[:, MEAN] == pytest.approx([1.0, 3.0])
    assert row.target == pytest.approx(np.full(5, 5.0))
    # The other channel's hourly means ride along for the ridge over the window: it read nothing.
    assert row.companions.tolist() == [2]
    assert row.means == pytest.approx(np.zeros((2, 1)))


def test_the_cells_and_their_context_are_written_for_the_tables_and_a_forecaster(
    tmp_path: Path,
) -> None:
    cells = [
        ScoredCell(
            window=4,
            axis="M1",
            channel=2,
            target_mean=1.5,
            losses=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            means=(1.4, 1.0, 1.2, 1.1, 1.6, 1.7),
            lag=None,
            history=(0.5, float("nan"), 1.25),
        )
    ]

    write_cells(tmp_path / "answers.csv", cells)
    write_context(tmp_path / "context.csv", cells, 3)

    with (tmp_path / "answers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(COLUMNS)
    assert (rows[0]["lag"], rows[0]["huber_ridge"], rows[0]["mean_decoder"]) == ("", "0.5", "1.4")
    with (tmp_path / "context.csv").open(newline="") as handle:
        context = list(csv.reader(handle))
    assert context == [
        ["window", "axis", "channel", "h01", "h02", "h03"],
        ["4", "M1", "2", "0.5", "", "1.25"],
    ]
    with pytest.raises(ValueError, match="3 hours"):
        write_context(tmp_path / "context.csv", cells, 2)


class FakeJoined:
    """Joined windows whose cells are given outright: an hour nobody read is a zero slot."""

    def __init__(self, cells: dict[int, np.ndarray]) -> None:
        self._cells = cells

    def __getitem__(self, position: int) -> Any:
        return SimpleNamespace(cells=self._cells[position])


def test_a_window_whose_history_holds_an_hour_nobody_read_is_incomplete() -> None:
    whole = np.ones((2, 4, FEATURES))
    gap = whole.copy()
    gap[:, 1, :] = 0.0  # the second hour of history, every channel
    own_hour_silent = whole.copy()
    own_hour_silent[:, -1, :] = 0.0  # the hidden hour itself, which is not history
    windows = WindowSet(
        cast(Any, FakeJoined({0: whole, 1: gap, 2: own_hour_silent})), np.arange(3), ("M1",) * 3, 0
    )

    assert [windows.complete(p) for p in range(3)] == [True, False, True]


def test_windows_are_drawn_per_axis_and_outside_the_ones_excluded() -> None:
    windows = WindowSet(
        cast(Any, FakeJoined({})),
        np.arange(10),
        ("M1",) * 6 + ("M2",) * 4,
        dropped=0,
    )

    drawn = windows.draw(3, seed=1)
    again = windows.draw(3, seed=1)
    fewer = windows.draw(3, seed=2, excluding=drawn.tolist())

    assert len(drawn) == 6
    assert list(drawn) == sorted(drawn)
    assert sum(1 for p in drawn if p < 6) == 3
    assert sum(1 for p in drawn if p >= 6) == 3
    assert list(again) == list(drawn)
    assert not set(fewer.tolist()) & set(drawn.tolist())
    assert sum(1 for p in fewer if p >= 6) == 1  # one window of the second axis was left
