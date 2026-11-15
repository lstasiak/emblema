"""The forecast's reading from the cells written: losses, ratios and the arm's level."""

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.history_forecast_reading import (
    ANSWERS,
    LOSS_COLUMNS,
    VERDICT_COLUMNS,
    CellSet,
    level_of,
    markdown,
    read,
    read_answers,
    threshold,
)


def answers_file(
    path: Path,
    decoder_scale: float,
    windows: range = range(6),
    ridge_failure: tuple[int, float] | None = None,
) -> Path:
    """Cells over ``windows`` and two channels: the ridge misses by 0.2, the decoder by a share.

    The decoder's Huber loss is ``decoder_scale`` of the ridge's on every cell, and each
    answer's mean is the target plus the root of twice its loss, so the mean-based measures
    agree with the five-value Huber inside the knee. ``ridge_failure`` names a window and the
    factor the ridge's loss takes there.
    """
    columns = [
        "window",
        "axis",
        "channel",
        "lag",
        "target_mean",
        *(f"huber_{a}" for a in ANSWERS),
        *(f"mean_{a}" for a in ANSWERS),
    ]
    losses = {
        "decoder": 0.02 * decoder_scale,
        "last_reading": 0.08,
        "median_of_history": 0.05,
        "same_phase": 0.03,
        "ridge": 0.02,
        "window_ridge": 0.015,
    }
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for w in windows:
            for c in (1, 2):
                target = float(w)
                here = dict(losses)
                if ridge_failure is not None and w == ridge_failure[0]:
                    here["ridge"] *= ridge_failure[1]
                writer.writerow(
                    [w, "M1", c, "", target]
                    + [here[a] for a in ANSWERS]
                    + [target + np.sqrt(2 * here[a]) for a in ANSWERS]
                )
    return path


def published_file(path: Path, windows: range = range(6)) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window", "axis", "channel", "mean_chronos2"])
        for w in windows:
            for c in (1, 2):
                writer.writerow([w, "M1", c, float(w) + 0.1])  # squared error 0.01 everywhere
    return path


def test_the_levels_read_the_arms_interval_against_one_and_the_least_gain() -> None:
    assert level_of(1.05, 1.2, 0.8) == "harm"
    assert level_of(0.9, 1.1, 0.8) == "noise: less than about the threshold either way"
    assert level_of(0.85, 0.95, 0.8) == "real but small"
    assert level_of(0.6, 0.8, 0.8) == "at least the least gain"


def test_the_threshold_takes_the_floored_spread_over_the_pretrainings() -> None:
    boot = np.zeros(100)  # no resampling variance

    spread, t = threshold(boot, [0.5, 0.5], floor=0.02)
    wider, t_wide = threshold(boot, [0.4, 0.6], floor=0.02)
    _, t_one = threshold(boot, [0.5], floor=0.02)

    assert spread == 0.02
    assert t == pytest.approx(2 * np.sqrt(0.02**2 / 4))
    assert wider == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert t_wide == pytest.approx(2 * np.sqrt(wider**2 / 4))
    assert t_one == pytest.approx(2 * np.sqrt(0.02**2 / 2))


def test_the_arm_is_the_mean_of_the_pretrainings_ratios_and_reads_a_level(
    tmp_path: Path,
) -> None:
    first = answers_file(tmp_path / "a.csv", 0.5)
    second = answers_file(tmp_path / "b.csv", 0.7)
    chronos = published_file(tmp_path / "c.csv")

    losses, verdicts = read(
        [("seed1", "unread", first), ("seed2", "unread", second)],
        [("unread", chronos)],
        delta=1.0,
        least_gain=0.8,
        floor=0.02,
        resamples=200,
        seed=1,
    )

    by = {(v["pretraining"], v["of"], v["against"], v["measure"]): v for v in verdicts}
    assert by[("seed1", "decoder", "ridge", "huber_five")]["ratio"] == pytest.approx(0.5)
    assert by[("seed2", "decoder", "ridge", "huber_five")]["ratio"] == pytest.approx(0.7)
    arm = by[("arm", "decoder", "ridge", "huber_five")]
    assert arm["ratio"] == pytest.approx(0.6)
    # The cells are identical across windows, so the resampling adds nothing: T is the spread's.
    spread = arm["spread"]
    assert isinstance(spread, float)
    assert spread == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert arm["threshold"] == pytest.approx(2 * np.sqrt(spread**2 / 4))
    assert arm["level"] == "at least the least gain"
    beside = by[("seed1", "decoder", "window_ridge", "huber_five")]
    assert beside["ratio"] == pytest.approx(0.01 / 0.015)
    assert by[("arm", "decoder", "window_ridge", "huber_five")]["level"] == "descriptive"
    # Every window carries the same ratio, so the median over windows is the ratio of means.
    median = by[("seed2", "decoder", "ridge", "huber_five_window_median")]
    assert median["ratio"] == pytest.approx(0.7)
    assert (median["low"], median["high"]) == pytest.approx((0.7, 0.7))
    assert median["level"] == "descriptive"
    # Against the published model on the mean's squared error: the decoder of seed 1 misses by
    # the root of 0.02, squared 0.02, over the published 0.01.
    assert by[("seed1", "decoder", "chronos2", "squared_mean")]["ratio"] == pytest.approx(2.0)
    low, high = (by[("seed1", "decoder", "ridge", "huber_five")][k] for k in ("low", "high"))
    assert isinstance(low, float)
    assert isinstance(high, float)
    assert low <= 0.5 <= high
    rows = {(r["pretraining"], r["answer"], r["measure"]): r for r in losses}
    assert rows[("seed1", "chronos2", "squared_mean")]["mean"] == pytest.approx(0.01)
    assert rows[("seed1", "ridge", "huber_mean")]["mean"] == pytest.approx(0.02)
    assert rows[("seed2", "decoder", "huber_five")]["cells"] == 12
    assert set(LOSS_COLUMNS) == set(losses[0])
    assert set(VERDICT_COLUMNS) == set(verdicts[0])


def test_the_median_over_windows_does_not_follow_one_window_the_bar_failed_on(
    tmp_path: Path,
) -> None:
    # On one of six windows the ridge misses fifty times worse: the ratio of means falls, the
    # median over windows stays where the typical window is.
    failed = answers_file(tmp_path / "a.csv", 0.9, ridge_failure=(3, 50.0))

    _, verdicts = read(
        [("seed1", "unread", failed)],
        [],
        delta=1.0,
        least_gain=0.8,
        floor=0.02,
        resamples=200,
        seed=1,
    )

    by = {(v["pretraining"], v["measure"]): v for v in verdicts if v["against"] == "ridge"}
    assert by[("seed1", "huber_five")]["ratio"] == pytest.approx(0.9 * 6 / (5 + 50))
    assert by[("seed1", "huber_five_window_median")]["ratio"] == pytest.approx(0.9)


def test_pretrainings_scored_on_different_windows_are_refused(tmp_path: Path) -> None:
    first = answers_file(tmp_path / "a.csv", 0.5)
    second = answers_file(tmp_path / "b.csv", 0.5, windows=range(1, 7))

    with pytest.raises(ValueError, match="different windows"):
        read(
            [("seed1", "unread", first), ("seed2", "unread", second)],
            [],
            delta=1.0,
            least_gain=0.8,
            floor=0.02,
            resamples=10,
            seed=1,
        )


def test_a_published_answer_that_misses_a_cell_is_refused(tmp_path: Path) -> None:
    cells = read_answers(answers_file(tmp_path / "a.csv", 0.5))

    with pytest.raises(ValueError, match="misses 2 cells"):
        CellSet.of(cells, delta=1.0, published={c.key: 0.0 for c in cells[:-2]})
    with pytest.raises(ValueError, match="no cell"):
        CellSet.of([], delta=1.0, published=None)


def test_the_table_shows_floats_to_four_places() -> None:
    text = markdown(("a", "b"), [{"a": 0.123456, "b": "x"}])

    assert text.splitlines()[-1] == "| 0.1235 | x |"
