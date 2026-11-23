"""The fault benchmark's excerpts: which turbines are read and which are never opened."""

from pathlib import Path

import numpy as np
import pytest

from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.normalisation import Normalisation
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.care_excerpts import (
    CONDITIONS,
    Event,
    Excerpt,
    HeldUnreadError,
    Hours,
    Sensors,
    cells_of,
    chosen,
    events_of,
    file_of,
    held_unread,
    read_events,
    robust_scores,
)

FARM_C = (2, 5, 12, 13, 14, 15, 16, 21, 23, 32, 33, 34, 35, 38, 42, 43, 44, 50, 52, 53, 55, 56)


def benchmark(root: Path) -> Path:
    """A farm B of nine turbines, one event each, and a farm A spelling its turbine's column."""
    for farm, column, turbines in (("A", "asset", (0, 10)), ("B", "asset_id", range(9))):
        folder = root / f"Wind Farm {farm}"
        (folder / "datasets").mkdir(parents=True)
        lines = [f"{column};event_id;event_label;event_start;event_start_id;event_end;event_end_id"]
        for turbine in turbines:
            label = "anomaly" if turbine % 2 else "normal"
            starts = f"2023-01-0{1 + turbine % 9} 00:00:00"
            lines.append(
                f"{turbine};{10 + turbine};{label};{starts};{100 + turbine};"
                f"2023-02-01 00:00:00;{900 + turbine}"
            )
        (folder / "event_info.csv").write_text("\n".join(lines) + "\n")
    return root


def test_the_turbines_held_unread_are_the_registered_ones() -> None:
    assert held_unread("A", (0, 10, 11, 13, 21)) == {0, 21}
    assert held_unread("B", (0, 2, 5, 6, 7, 11, 12, 13, 14)) == {0, 5, 6, 14}
    assert held_unread("C", FARM_C) == {13, 21, 23, 32, 33, 38, 42, 50, 52, 53, 56}


def test_a_farms_events_are_read_under_either_spelling_of_the_turbines_column(
    tmp_path: Path,
) -> None:
    root = benchmark(tmp_path)

    assert events_of(root, "A")[1] == Event(
        farm="A",
        event=20,
        turbine=10,
        anomaly=False,
        first_row=110,
        last_row=910,
    )
    assert [event.anomaly for event in events_of(root, "B")][:3] == [False, True, False]


def test_the_events_read_leave_out_the_turbines_held_unread(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    held = held_unread("B", range(9))

    read = read_events(root, farms=("B",))

    assert len(held) == 4
    assert {event.turbine for event in read} == set(range(9)) - held


def test_a_file_of_a_turbine_held_unread_is_refused(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    held = next(
        event for event in events_of(root, "B") if event.turbine in held_unread("B", range(9))
    )

    with pytest.raises(HeldUnreadError, match="is held unread"):
        file_of(root, held)


def test_an_event_naming_another_turbine_than_the_farms_table_is_refused(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    held = held_unread("B", range(9))
    of_held = next(event for event in events_of(root, "B") if event.turbine in held)
    read = next(turbine for turbine in range(9) if turbine not in held)

    # The held turbine's excerpt, handed in under a read turbine's name.
    with pytest.raises(ValueError, match="holds no excerpt"):
        file_of(root, Event("B", of_held.event, read, True, 0, 1))


def test_a_run_reads_the_farms_excerpts_or_among_them_the_ones_named() -> None:
    events = [Event(farm, number, 1, True, 0, 1) for farm in "AB" for number in (3, 4)]

    assert [event.event for event in chosen(events, ["B"], [])] == [3, 4]
    assert [(e.farm, e.event) for e in chosen(events, ["A", "B"], [["A", "4"]])] == [("A", 4)]


def test_an_excerpt_named_that_is_not_read_is_refused() -> None:
    events = [Event("A", 3, 1, True, 0, 1)]

    with pytest.raises(ValueError, match="not among the excerpts read: B 3"):
        chosen(events, ["A", "B"], [["B", "3"]])
    with pytest.raises(ValueError, match="not among the excerpts read: A 3"):
        chosen(events, ["B"], [["A", "3"]])


def test_a_read_turbines_file_is_named_by_its_event(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    event = read_events(root, farms=("B",))[0]

    assert file_of(root, event) == root / "Wind Farm B" / "datasets" / f"{event.event}.csv"


def test_the_excerpt_lying_inside_a_fault_is_left_out_of_rankings() -> None:
    inside = Event("B", 87, 7, False, 52475, 54779)
    other = Event("B", 27, 7, True, 52619, 61403)

    assert not inside.ranked
    assert other.ranked


# --- an excerpt on its clock hours ------------------------------------------------------------

START = np.datetime64("2022-03-01T00:00:00")


def excerpt(
    values: np.ndarray,
    *,
    fitted_rows: int,
    event_rows: tuple[int, int],
    status: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    starts: np.datetime64 = START,
    farm: str = "B",
) -> Excerpt:
    """Rows every ten minutes from ``starts``, the first ``fitted_rows`` the fitted year."""
    rows = len(values)
    kept = np.ones(rows, dtype=np.bool_) if keep is None else keep
    stamps = (starts + np.arange(rows) * np.timedelta64(600, "s"))[kept]
    ids = np.arange(rows, dtype=np.int64)[kept]
    return Excerpt.of(
        Event(farm, 1, 3, True, event_rows[0], event_rows[1]),
        [f"sensor_{column}_avg" for column in range(values.shape[1])],
        stamps=stamps.astype("datetime64[s]"),
        ids=ids,
        training=["train" if row < fitted_rows else "prediction" for row in ids],
        status=(np.zeros(rows, dtype=np.int64) if status is None else status)[kept],
        values=values[kept],
    )


def level(rows: int, columns: int = 2) -> np.ndarray:
    return 20.0 + np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) / 100.0


def test_the_detectors_channels_leave_out_angles_and_counters() -> None:
    rows = [
        {"sensor_name": "sensor_0", "unit": "kWh", "is_angle": "False", "is_counter": "True"},
        {"sensor_name": "sensor_8", "unit": "�C", "is_angle": "False", "is_counter": "False"},
        {"sensor_name": "sensor_10", "unit": "�", "is_angle": "True", "is_counter": "False"},
        {"sensor_name": "sensor_25", "unit": "rpm", "is_angle": "False", "is_counter": "False"},
        {"sensor_name": "sensor_32", "unit": "�C", "is_angle": "False", "is_counter": "False"},
        {"sensor_name": "wind_speed_61", "unit": "m/s", "is_angle": "False", "is_counter": "False"},
        {"sensor_name": "power_62", "unit": "kW", "is_angle": "False", "is_counter": "False"},
    ]
    averages = [f"{row['sensor_name']}_avg" for row in rows]

    sensors = Sensors.described("B", rows, averages)

    assert "sensor_0_avg" not in sensors.channels
    assert "sensor_10_avg" not in sensors.channels
    assert "sensor_25_avg" in sensors.channels
    # The outside temperature is a condition; the model answers the bearing's alone.
    assert sensors.temperatures == ("sensor_32_avg",)
    assert sensors.conditions == tuple(f"{name}_avg" for name in CONDITIONS["B"])


def test_a_farm_without_one_of_its_conditions_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no average of"):
        Sensors.described("B", [], ["power_62_avg"])


def test_a_row_with_half_its_averages_at_nought_is_a_dropped_row() -> None:
    values = level(12, columns=4)
    values[3] = 0.0
    values[5, :2] = 0.0
    values[7, 0] = 0.0

    read = excerpt(values, fitted_rows=6, event_rows=(6, 11))

    assert np.isnan(read.values[3]).all()
    assert np.isnan(read.values[5]).all()
    assert read.values[7, 0] == 0.0


def test_rows_out_of_time_order_are_refused() -> None:
    read = excerpt(level(12), fitted_rows=6, event_rows=(6, 11))

    with pytest.raises(ValueError, match="not in time order"):
        Excerpt.of(
            read.event,
            read.columns,
            stamps=read.stamps[::-1],
            ids=read.ids,
            training=["train"] * 12,
            status=np.zeros(12, dtype=np.int64),
            values=read.values,
        )


def test_a_fitted_row_after_a_scored_one_is_refused() -> None:
    read = excerpt(level(12), fitted_rows=6, event_rows=(6, 11))

    with pytest.raises(ValueError, match="fitted rows after scored"):
        Excerpt.of(
            read.event,
            read.columns,
            stamps=read.stamps,
            ids=read.ids,
            training=["train"] * 5 + ["prediction"] + ["train"] * 6,
            status=np.zeros(12, dtype=np.int64),
            values=read.values,
        )


def test_only_statuses_nought_and_two_are_normal_operation() -> None:
    status = np.array([0, 1, 2, 3, 4, 5] * 2)

    read = excerpt(level(12), fitted_rows=6, event_rows=(6, 11), status=status)

    assert read.normal.tolist() == [True, False, True, False, False, False] * 2


def test_the_first_farms_scored_rows_are_read_whatever_their_status() -> None:
    status = np.zeros(24, dtype=np.int64)
    status[3:5] = 4
    status[14:20] = 4

    first = excerpt(level(24), fitted_rows=12, event_rows=(12, 23), status=status, farm="A")
    other = excerpt(level(24), fitted_rows=12, event_rows=(12, 23), status=status, farm="B")
    third = excerpt(level(24), fitted_rows=12, event_rows=(12, 23), status=status, farm="C")

    # The fitted year is read by its status in every farm; the status itself is kept.
    assert not first.read[3:5].any()
    assert first.read[12:].all()
    assert np.array_equal(first.normal, other.normal)
    assert np.array_equal(other.read, other.normal)
    assert np.array_equal(third.read, third.normal)
    assert Hours.of(first).whole.tolist() == [False, True, True, True]
    assert Hours.of(other).whole.tolist() == [False, True, False, False]


def test_an_hour_is_whole_with_six_rows_all_of_normal_operation() -> None:
    status = np.zeros(36, dtype=np.int64)
    status[8] = 4
    keep = np.ones(36, dtype=np.bool_)
    keep[14] = False

    hours = Hours.of(
        excerpt(level(36), fitted_rows=24, event_rows=(24, 35), status=status, keep=keep)
    )

    assert hours.count == 6
    assert hours.whole.tolist() == [True, False, False, True, True, True]


def test_hours_a_file_skips_keep_their_place_on_the_axis() -> None:
    keep = np.ones(60, dtype=np.bool_)
    keep[12:36] = False

    hours = Hours.of(excerpt(level(60), fitted_rows=48, event_rows=(48, 59), keep=keep))

    assert hours.count == 10
    assert hours.whole.tolist() == [True, True] + [False] * 4 + [True] * 4
    assert hours.fitted == 8
    assert (hours.first, hours.last) == (8, 9)


def test_the_hour_a_scored_row_falls_in_is_not_fitted() -> None:
    # The file starts twenty minutes past the hour, so the scored window does too.
    starts = START + np.timedelta64(1200, "s")

    hours = Hours.of(excerpt(level(36), fitted_rows=18, event_rows=(20, 35), starts=starts))

    assert hours.of_row[:5].tolist() == [0, 0, 0, 0, 1]
    assert hours.fitted == 3
    assert hours.first == 3
    assert not hours.whole[0]


def test_an_event_starting_inside_the_fitted_year_is_refused() -> None:
    with pytest.raises(ValueError, match="starts inside the fitted year"):
        Hours.of(excerpt(level(36), fitted_rows=24, event_rows=(12, 35)))


def test_an_event_with_no_row_in_the_file_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no row of its event"):
        Hours.of(excerpt(level(36), fitted_rows=24, event_rows=(400, 500)))


def test_a_channel_is_read_in_an_hour_only_with_all_six_of_its_readings() -> None:
    values = level(18)
    values[7, 0] = np.nan
    status = np.zeros(18, dtype=np.int64)
    status[13] = 3
    read = excerpt(values, fitted_rows=12, event_rows=(12, 17), status=status)

    readings = Hours.of(read).readings(read.values)

    assert readings.shape == (3, 2, 6)
    assert readings[0, 1].tolist() == values[:6, 1].tolist()
    assert np.isnan(readings[1, 0]).all()
    assert not np.isnan(readings[1, 1]).any()
    assert np.isnan(readings[2]).all()


def test_the_share_of_an_hour_not_of_normal_operation_counts_its_rows() -> None:
    status = np.zeros(18, dtype=np.int64)
    status[6:9] = 4
    keep = np.ones(18, dtype=np.bool_)
    keep[12:18] = False
    keep[17] = True
    read = excerpt(level(18), fitted_rows=6, event_rows=(6, 17), status=status, keep=keep)

    assert Hours.of(read).abnormal_share(read.normal).tolist() == [0.0, 0.5, 0.0]


def test_an_hour_the_file_skips_is_wholly_out_of_normal_operation() -> None:
    keep = np.ones(24, dtype=np.bool_)
    keep[6:12] = False
    read = excerpt(level(24), fitted_rows=18, event_rows=(18, 23), keep=keep)

    assert Hours.of(read).abnormal_share(read.normal).tolist() == [0.0, 1.0, 0.0, 0.0]


def test_readings_are_scored_as_a_publication_scores_a_channel() -> None:
    draws = np.random.default_rng(1)
    readings = draws.normal(50.0, 4.0, size=(40, 3, 6))
    readings[:, 1] = 7.0
    readings[::9, 1] = 9.0
    readings[:, 2] = 3.0

    scored = robust_scores(readings, fitted=30)

    for channel in range(3):
        year = readings[:30, channel].ravel()
        median = float(np.median(year))
        statistics = ChannelStatistics(
            count=len(year),
            mean=float(year.mean()),
            std=float(year.std()),
            normalisation=Normalisation.ROBUST,
            median=median,
            mad=float(np.median(np.abs(year - median))),
            bound=100.0,
        )
        assert scored[35, channel, 2] == pytest.approx(
            statistics.normalise(float(readings[35, channel, 2]))
        )


def test_a_score_is_held_to_the_bound() -> None:
    readings = np.random.default_rng(3).normal(size=(40, 1, 6))
    readings[35] = 1e9

    assert robust_scores(readings, fitted=30).max() == 100.0


def test_every_reading_is_a_span_of_its_own_with_a_count_of_one() -> None:
    readings = level(12).reshape(2, 6, 2).transpose(0, 2, 1).copy()
    readings[1, 0] = np.nan

    (cells,) = cells_of(readings)

    names = BinnedTokens.FEATURE_NAMES
    assert cells.shape == (2, 2, 6, len(names))
    assert cells[0, 0, :, names.index("presence")].tolist() == [1.0] * 6
    assert cells[1, 0, :, names.index("presence")].tolist() == [0.0] * 6
    assert set(cells[0, :, :, names.index("log_count")].ravel()) == {np.log1p(1.0)}
    for name in ("mean", "minimum", "maximum", "last"):
        assert np.array_equal(cells[0, :, :, names.index(name)], readings[0])


def test_the_cells_come_in_chunks_of_hours() -> None:
    readings = np.zeros((5, 1, 6))

    assert [len(chunk) for chunk in cells_of(readings, chunk=2)] == [2, 2, 1]
