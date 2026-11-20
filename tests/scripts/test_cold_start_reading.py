"""The cold-start reading: files joined by cell, the geometric ratio, the bar and its levels."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.cold_start_reading import (
    Day,
    comparison_rows,
    day_weights,
    geometric,
    joined,
    loss_rows,
    pooled,
)


def write(path: Path, header: str, *rows: str) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_the_files_are_joined_by_cell_and_a_blank_answer_reads_missing(tmp_path: Path) -> None:
    answers = write(
        tmp_path / "answers.csv",
        "window,axis,channel,lag,target_mean,mean_last_hour,mean_ridge",
        "5,M,0,,1.0,0.5,",
        "5,M,1,2,2.0,1.5,1.9",
    )
    local = write(
        tmp_path / "local.csv", "window,axis,channel,mean_auto_ets", "5,M,1,2.1", "5,M,0,1.1"
    )

    day = joined("M", "7", [answers, local])

    assert day.keys == ((5, "M", 0), (5, "M", 1))
    assert day.answers["auto_ets"].tolist() == [1.1, 2.1]
    assert np.isnan(day.answers["ridge"][0])
    assert day.covered(["last_hour", "ridge"]).tolist() == [False, True]


def test_files_of_other_cells_or_an_answer_twice_are_refused(tmp_path: Path) -> None:
    answers = write(
        tmp_path / "answers.csv", "window,axis,channel,target_mean,mean_a", "5,M,0,1.0,0.5"
    )
    other = write(tmp_path / "other.csv", "window,axis,channel,mean_b", "6,M,0,1.0")
    twice = write(tmp_path / "twice.csv", "window,axis,channel,mean_a", "5,M,0,1.0")

    with pytest.raises(ValueError, match="other cells"):
        joined("M", "7", [answers, other])
    with pytest.raises(ValueError, match="two files"):
        joined("M", "7", [answers, twice])


def test_the_ratio_is_against_the_bar_led_least_in_every_draw() -> None:
    numerator = np.array([1.0, 1.0, 1.0, 1.0])
    weak, strong = np.full(4, 4.0), np.full(4, 2.0)
    draws = day_weights(np.array([0, 1, 30, 50]), 200, seed=0)

    ratio, low, high, best = geometric(numerator, [weak, strong], draws)

    assert ratio == pytest.approx(0.5)
    assert best == 1
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)
    with pytest.raises(ValueError, match="missed by nothing"):
        geometric(np.array([0.0, 1.0]), [np.ones(2)], draws[:, :2])


def day_of(days: int, first: int = 1) -> Day:
    keys = tuple((w, "M", c) for w in range(first, first + 6) for c in range(2))
    target = np.zeros(len(keys))
    errors = {
        "chronos2_47": 0.5,
        "chronos2_own": 0.4,
        "last_hour": 1.0,
        "auto_ets": 0.8,
        "ridge": 0.6,
    }
    answers = {name: np.full(len(keys), error) for name, error in errors.items()}
    answers["ridge"][0] = np.nan
    return Day("M", str(days), keys, target, answers)


def test_an_answer_missing_a_cell_stays_out_of_the_bar_and_only_verdicts_read_a_level() -> None:
    rows = comparison_rows(
        day_of(7),
        verdicts={("chronos2_47", "classical")},
        least_gain=0.8,
        resamples=200,
        seed=1,
    )

    by = {(str(r["of"]), str(r["against"]).split(":")[0]): r for r in rows}
    first = by[("chronos2_47", "best of 2")]
    assert first["against"] == "best of 2: auto_ets"
    assert first["ratio"] == pytest.approx((0.5 / 0.8) ** 2)
    assert first["level"] == "at least the least gain"
    assert first["cell_median"] == pytest.approx((0.5 / 0.8) ** 2)
    assert by[("chronos2_own", "chronos2_47")]["level"] == "descriptive"
    assert not any(r["of"] == "chronos2_analog" for r in rows)


def test_each_answer_reads_its_own_geometric_loss_on_the_cells_it_answers() -> None:
    rows = {r["answer"]: r for r in loss_rows(day_of(7), resamples=100, seed=1)}

    assert rows["auto_ets"]["geometric_mean"] == pytest.approx(0.64)
    assert rows["ridge"]["cells"] == 11


def test_pooled_days_hold_every_cell_once_and_an_answer_one_day_lacks_reads_missing() -> None:
    first = day_of(2)
    early = replace(first, answers={k: v for k, v in first.answers.items() if k != "ridge"})
    late = day_of(7, first=10)

    both = pooled([early, late])

    assert both.days == "2-7"
    assert len(both.keys) == 24
    assert np.isnan(both.answers["ridge"][:12]).all()
    assert not both.covered(["ridge"]).all()
    with pytest.raises(ValueError, match="same cell"):
        pooled([late, late])


def test_the_draws_take_or_leave_a_days_hours_together() -> None:
    hours = np.array([0, 5, 23, 24, 30, 100])

    weights = day_weights(hours, 500, seed=2)

    assert weights.shape == (500, 6)
    assert (weights[:, 0] == weights[:, 1]).all()
    assert (weights[:, 1] == weights[:, 2]).all()
    assert (weights[:, 3] == weights[:, 4]).all()
    assert (weights.sum(axis=1) > 0).all()
    assert weights[:, 5].mean() == pytest.approx(1.0, abs=0.15)
