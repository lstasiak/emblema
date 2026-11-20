"""The channel-local detector fitted on a mission's first days and read on later months."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from scripts.channel_local_cold_start import (
    WINDOW_COLUMNS,
    Departures,
    Reading,
    Windows,
    answered,
    curve_rows,
    drawn_f,
    parse_arguments,
    scored_side,
    side_of,
)
from scripts.telemetry_channel_local_report import Arm
from scripts.telemetry_verdict_measure_report import Tally


def write_windows(path: Path, rows: Sequence[tuple[object, ...]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(WINDOW_COLUMNS)
        writer.writerows(rows)
    return path


def some_windows(path: Path) -> Windows:
    rows = [(i, "pool", "M/2000-01", i, float(i + 1), 0.0) for i in range(200)]
    rows += [(200 + i, "read", "M/2007-01", 500 + i, 9000.0 + i, float(i == 3)) for i in range(4)]
    return Windows.read(write_windows(path / "windows.csv", rows))


def test_the_first_days_are_the_fitting_windows_with_their_widest_context_inside_the_mission(
    tmp_path: Path,
) -> None:
    windows = some_windows(tmp_path)

    assert windows.fitting_read(4).tolist() == list(range(71, 96))
    assert windows.fitting_read().tolist() == list(range(71, 200))
    with pytest.raises(ValueError, match="72 hours before"):
        windows.fitting_read(2)
    assert windows.validation().tolist() == [200, 201, 202, 203]
    assert windows.read_of(windows.validation(), {("M/2007-01", 501)}).tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert windows.targets[203] == 1


def test_windows_out_of_their_laid_order_or_on_another_side_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="numbered"):
        Windows.read(write_windows(tmp_path / "a.csv", [(1, "pool", "M/a", 0, 1.0, 0.0)]))
    with pytest.raises(ValueError, match="lies on"):
        Windows.read(write_windows(tmp_path / "b.csv", [(0, "test", "M/a", 0, 1.0, 0.0)]))


def test_the_scored_side_runs_by_month_and_place(tmp_path: Path) -> None:
    rows = [(0, "pool", "M/a", 0, 1.0, 0.0), (1, "pool", "M/a", 1, 2.0, 0.0)]
    rows += [(2, "read", "M/c", 9, 10.0, 0.0), (3, "read", "M/b", 7, 8.0, 2.0)]
    rows += [(4, "read", "M/b", 6, 7.0, 0.0)]
    windows = Windows.read(write_windows(tmp_path / "w.csv", rows))

    side = side_of(windows, np.asarray([2, 3, 4], dtype=np.int64), np.asarray([0.3, 0.2, 0.1]))

    assert side.units.tolist() == [0, 0, 1]
    assert side.positions.tolist() == [6, 7, 9]
    assert side.events.tolist() == [0, 2, 0]
    assert side.scores.tolist() == [0.1, 0.2, 0.3]


def test_the_f_score_over_every_month_once_is_the_tallys_own() -> None:
    tallied = Tally(
        events=np.asarray([2, 1, 0], dtype=np.int64),
        found=np.asarray([1, 1, 0], dtype=np.int64),
        false_runs=np.asarray([3, 0, 4], dtype=np.int64),
        normal=np.asarray([100, 90, 80], dtype=np.int64),
        quiet=np.asarray([97, 90, 75], dtype=np.int64),
    )

    once, drawn = drawn_f(tallied, np.asarray([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]]))
    replicated = Tally(
        **{
            name: np.asarray([values[0], values[0], values[2]], dtype=np.int64)
            for name, values in vars(tallied).items()
        }
    )

    assert once == pytest.approx(tallied.f_score())
    assert drawn == pytest.approx(replicated.f_score())


def laid_departures(directory: Path, summaries: np.ndarray) -> Departures:
    rng = np.random.default_rng(3)
    windows, rows = summaries.shape[:2]
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "summaries.npy", summaries)
    np.save(directory / "hour-6h-centre.npy", rng.normal(size=(windows, rows, 5)))
    np.save(directory / "hour-6h-spread.npy", rng.uniform(0.0, 0.2, size=(windows, rows, 5)))
    np.save(directory / "hour-6h-loaded.npy", np.full((windows, rows), 5, dtype=np.int64))
    return Departures(directory, np.zeros((windows, rows, 6, 9)))


def test_the_floor_and_the_alarm_read_the_fitting_windows_alone(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    summaries = rng.normal(size=(200, 3, 5))
    rows = [
        (i, "pool" if i < 150 else "read", "M/a" if i < 150 else "M/b", i, float(i + 1), 0.0)
        for i in range(200)
    ]
    windows = Windows.read(write_windows(tmp_path / "w.csv", rows))
    fit, scored = np.arange(100), np.arange(150, 200)

    def answers(values: np.ndarray, name: str) -> np.ndarray:
        departures = laid_departures(tmp_path / name, values)
        return answered(
            departures,
            windows,
            arm=Arm("hour", 6),
            detector="max",
            fit=fit,
            scored=scored,
            present=departures.present(),
        )

    first = answers(summaries, "first")
    elsewhere = summaries.copy()
    elsewhere[100:150] = rng.normal(size=(50, 3, 5)) * 50
    fitted = summaries.copy()
    fitted[:100] = rng.normal(size=(100, 3, 5)) * 50

    assert np.array_equal(first, answers(elsewhere, "elsewhere"))
    assert not np.array_equal(first, answers(fitted, "fitted"))
    assert np.mean(first > 0.0) < 0.2


def test_the_days_are_distinct_and_rising() -> None:
    common = ["curve", "--run", "r", "--cells", "c", "--windows", "w", "--out", "o", "--days"]

    assert parse_arguments([*common, "2", "7"]).days == [2, 7]
    with pytest.raises(SystemExit):
        parse_arguments([*common, "7", "2"])


def test_each_month_is_answered_whole_before_the_windows_not_read_are_left_out(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(5)
    rows = [(i, "pool", "M/a", i, float(i + 1), 0.0) for i in range(100)]
    rows += [(100 + i, "read", "M/b", 100 + i, float(101 + i), 0.0) for i in range(40)]
    windows = Windows.read(write_windows(tmp_path / "w.csv", rows))
    departures = laid_departures(tmp_path / "d", rng.normal(size=(140, 3, 5)))
    dropped = {("M/b", 100 + i) for i in range(10)}
    arm, fit, present = Arm("hour", 6), np.arange(100), departures.present()

    side = scored_side(
        departures, windows, arm=arm, detector="max", fit=fit, dropped=dropped, present=present
    )
    whole, alone = (
        answered(
            departures, windows, arm=arm, detector="max", fit=fit, scored=scored, present=present
        )
        for scored in (np.arange(100, 140), np.arange(110, 140))
    )

    assert side.positions.tolist() == list(range(110, 140))
    assert np.array_equal(side.scores, whole[10:])
    assert not np.array_equal(side.scores, alone)


def test_a_fitting_side_scoring_without_spread_in_every_month_is_refused(tmp_path: Path) -> None:
    rows = [(i, "pool", "M/a", i, float(i + 1), 0.0) for i in range(100)]
    rows += [(100 + i, "read", "M/b", 100 + i, float(101 + i), 0.0) for i in range(20)]
    windows = Windows.read(write_windows(tmp_path / "w.csv", rows))
    departures = laid_departures(tmp_path / "d", np.zeros((120, 3, 5)))
    np.save(tmp_path / "d" / "hour-6h-centre.npy", np.zeros((120, 3, 5)))

    with pytest.raises(ValueError, match="no fitting month"):
        answered(
            departures,
            windows,
            arm=Arm("hour", 6),
            detector="max",
            fit=np.arange(100),
            scored=np.arange(100, 120),
            present=departures.present(),
        )


def test_a_fitting_side_read_against_itself_differs_by_nought() -> None:
    tallied = Tally(
        events=np.asarray([2, 0], dtype=np.int64),
        found=np.asarray([1, 0], dtype=np.int64),
        false_runs=np.asarray([3, 1], dtype=np.int64),
        normal=np.asarray([50, 60], dtype=np.int64),
        quiet=np.asarray([48, 59], dtype=np.int64),
    )
    reading = {
        ("max", "hour-6h"): Reading(at_one=tallied, chance=np.asarray([0.4, 0.0]), own=tallied)
    }

    [row] = curve_rows("all", 10, reading, reading, np.ones((5, 2)))

    assert row["excess_at_1"] == "0.60"
    assert (row["excess_at_1_less_all"], row["excess_low"], row["excess_high"]) == ("0.00",) * 3
    assert (row["f_own_less_all"], row["f_low"], row["f_high"]) == ("0.0000",) * 3


def test_a_month_without_spread_among_others_is_read_as_the_verdict_reads_it(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(6)
    rows = [(i, "pool", "M/a" if i < 50 else "M/b", i, float(i + 1), 0.0) for i in range(100)]
    rows += [(100 + i, "read", "M/c", 100 + i, float(101 + i), 0.0) for i in range(20)]
    windows = Windows.read(write_windows(tmp_path / "w.csv", rows))
    summaries = rng.normal(size=(120, 3, 5))
    departures = laid_departures(tmp_path / "d", summaries)
    centre = np.load(tmp_path / "d" / "hour-6h-centre.npy")
    centre[:50] = summaries[:50]
    np.save(tmp_path / "d" / "hour-6h-centre.npy", centre)

    answers = answered(
        departures,
        windows,
        arm=Arm("hour", 6),
        detector="max",
        fit=np.arange(100),
        scored=np.arange(100, 120),
        present=departures.present(),
    )

    assert np.isfinite(answers).all()
