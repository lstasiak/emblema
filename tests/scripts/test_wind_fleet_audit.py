from typing import Any

import numpy as np
import pytest

from emblema.catalog.adapters.readers.greenbyte import GreenbyteCorpusReader, Signal, WindFarm
from scripts import wind_fleet_audit
from scripts.wind_fleet_audit import (
    coverage_rows,
    held_out,
    hourly_rows,
    implausible_rows,
    independent_signals,
    most_rows_of_an_instant,
    one_per_instant,
    read_farm,
    shared_time_groups,
    turbine_rows,
    value_row,
)
from tests.support.corpora import sample

pd = pytest.importorskip("pandas")
# pandas ships no type information, so a frame is whatever it hands back.
Frame = Any
POWER = Signal("Power", "kW")


def frame(stamps: list[str], **columns: list[float]) -> Frame:
    return pd.DataFrame(columns, index=pd.to_datetime(stamps))


def hourly(values: dict[str, np.ndarray]) -> Frame:
    """A turbine with one reading an hour of the signals named and none of the others."""
    hours = pd.date_range("2016-01-01", periods=len(next(iter(values.values()))), freq="1h")
    columns = {
        signal.name: values.get(signal.name, np.full(len(hours), np.nan))
        for signal in GreenbyteCorpusReader.SIGNALS
    }
    return pd.DataFrame(columns, index=hours)


def test_rows_of_one_instant_are_one_row_holding_each_columns_value() -> None:
    nan = float("nan")
    table = frame(
        ["2016-03-01 00:00", "2016-03-01 00:00", "2016-03-01 00:10"],
        Power=[1.0, nan, 2.0],
        Oil=[nan, 40.0, 41.0],
    )

    kept, disputed = one_per_instant(table)

    assert kept.to_dict("list") == {"Power": [1.0, 2.0], "Oil": [40.0, 41.0]}
    assert disputed == 0


def test_an_instant_whose_rows_disagree_is_counted() -> None:
    table = frame(["2016-03-01 00:00", "2016-03-01 00:00"], Power=[1.0, 1.5])

    assert one_per_instant(table)[1] == 1


def test_the_repeats_of_a_cumulative_table_are_counted_over_its_every_row() -> None:
    stamps = np.asarray(
        ["2016-03-01T00:00", "2016-03-01T00:00", "2016-03-01T00:10", "2016-03-01T00:00"],
        dtype="datetime64[s]",
    )

    assert most_rows_of_an_instant(stamps) == 3
    assert most_rows_of_an_instant(stamps[:0]) == 0


@pytest.mark.parametrize(
    ("corpus", "farm", "turbines", "tables"),
    [
        ("kelmarsh", GreenbyteCorpusReader.KELMARSH, ("1", "2"), 2),
        ("penmanshiel", GreenbyteCorpusReader.PENMANSHIEL, ("01", "15"), 2),
    ],
)
def test_the_audit_counts_the_readings_the_reader_counts(
    corpus: str, farm: WindFarm, turbines: tuple[str, ...], tables: int
) -> None:
    pytest.importorskip("pyarrow")
    root = sample(corpus)

    frames, facts = read_farm(root, farm.name)

    described = GreenbyteCorpusReader(root, farm, turbines).describe().content
    assert sorted(frames) == list(turbines)
    assert sum(int(frame.notna().sum().sum()) for frame in frames.values()) == (
        described.observation_count
    )
    assert len(facts) == tables
    assert all((fact.rows, fact.most_rows_of_an_instant) == (fact.instants, 1) for fact in facts)


def test_coverage_is_the_share_of_the_turbines_instants_that_hold_a_signal() -> None:
    nan = float("nan")
    one = frame(["2016-03-01", "2016-03-02", "2017-03-01"], Power=[1.0, nan, 1.0])
    other = frame(["2016-03-01", "2016-03-02"], Power=[1.0, 1.0])

    assert coverage_rows({"1": one, "2": other}) == [
        {"signal": "Power", "year": 2016, "share": 0.75},
        {"signal": "Power", "year": 2017, "share": 1.0},
    ]


def test_a_signals_readings_are_scored_against_its_median_and_robust_deviation() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0, 5.0])

    row = value_row(POWER, values)

    # The median is 5 and the median absolute deviation 2, so the robust deviation is 2.9652 and
    # the one reading at 1,000 lies 336 of them away.
    assert (row["median"], row["minimum"], row["maximum"]) == (5.0, 1.0, 1000.0)
    assert row["robust_deviation"] == pytest.approx(2.9652)
    assert row["share_beyond_100"] == pytest.approx(1 / 11)


def test_a_signal_with_no_robust_spread_is_scored_against_its_standard_deviation() -> None:
    # Nought but for one reading in a thousand: no robust spread, a standard deviation of 9.5.
    values = np.asarray([0.0] * 999 + [300.0])

    row = value_row(POWER, values)

    assert row["robust_deviation"] == 0.0
    assert row["share_beyond_100"] == 0.0
    assert value_row(POWER, np.asarray([0.0] * 99_999 + [300.0]))["share_beyond_100"] == 1e-5


def test_a_constant_signal_lies_nowhere() -> None:
    assert value_row(POWER, np.ones(5))["share_beyond_100"] == 0.0


def test_readings_outside_a_units_physical_range_are_counted_with_the_hours_they_touch() -> None:
    oil = np.full(48, 55.0)
    oil[[3, 4, 30]] = [-273.2, -273.2, 215.0]
    wind = np.full(48, 6.0)
    frames = {"1": hourly({"Gear oil temperature": oil, "Wind speed": wind, "Power": wind})}

    rows = {row["signal"]: row for row in implausible_rows(frames)}

    assert rows["Gear oil temperature"] == {
        "signal": "Gear oil temperature",
        "unit": "°C",
        "readings": 48,
        "below": 2,
        "above": 1,
        "commonest_value": -273.2,
        "readings_at_it": 2,
        "hours": 48,
        "hours_touched": 3,
    }
    assert (rows["Wind speed"]["below"], rows["Wind speed"]["commonest_value"]) == (0, "")
    # Power is in no unit with a range stated here.
    assert "Power" not in rows


def test_hourly_changes_are_read_between_turbines_and_against_a_day_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wind_fleet_audit, "LEAST_SHARED", 10)
    rng = np.random.default_rng(1)
    wind = rng.normal(size=600).cumsum()
    daily = np.sin(np.arange(600) * 2 * np.pi / 24)
    frames = {
        "1": hourly({"Wind speed": wind, "Nacelle ambient temperature": daily}),
        "2": hourly({"Wind speed": wind + 0.01 * rng.normal(size=600), "Power": wind}),
        "3": hourly({"Wind speed": wind, "Power": rng.normal(size=600).cumsum()}),
    }

    rows = {row["signal"]: row for row in hourly_rows(frames)}

    speed, power = rows["Wind speed"], rows["Power"]
    assert (speed["turbines"], power["turbines"]) == (3, 2)
    assert speed["changes_between_turbines_median"] == pytest.approx(1.0, abs=0.01)
    assert power["changes_between_turbines_median"] == pytest.approx(0.0, abs=0.15)
    assert rows["Nacelle ambient temperature"]["change_a_day_before"] == pytest.approx(1.0)
    assert speed["change_a_day_before"] == pytest.approx(0.0, abs=0.2)
    assert speed["level_an_hour_before"] == pytest.approx(1.0, abs=0.1)


def test_signals_that_move_together_count_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wind_fleet_audit, "LEAST_SHARED", 10)
    rng = np.random.default_rng(2)
    shared = rng.normal(size=400).cumsum()
    temperatures = [s.name for s in GreenbyteCorpusReader.SIGNALS if s.unit == "°C"]
    together = hourly(dict.fromkeys(temperatures, shared))
    apart = hourly({name: rng.normal(size=400).cumsum() for name in temperatures})

    one, many = independent_signals({"1": together, "2": apart})

    assert one == {"turbine": "1", "temperatures": len(temperatures), "behave_as": 1.0}
    assert (many["turbine"], many["temperatures"]) == ("2", len(temperatures))
    assert many["behave_as"] == pytest.approx(len(temperatures), rel=0.5)


def test_a_turbines_span_and_steps_are_read_off_its_instants() -> None:
    nan = float("nan")
    stamps = ["2016-01-23 12:10", "2016-01-23 12:20", "2016-05-03 09:40", "2016-05-03 09:50"]
    turbine = pd.DataFrame(
        {
            signal.name: [nan, nan, 40.0, 41.0] if signal.unit == "°C" else [1.0, 1.0, 1.0, nan]
            for signal in GreenbyteCorpusReader.SIGNALS
        },
        index=pd.to_datetime(stamps),
    )
    temperatures = sum(signal.unit == "°C" for signal in GreenbyteCorpusReader.SIGNALS)

    assert turbine_rows({"1": turbine}) == [
        {
            "turbine": "1",
            "instants": 4,
            "readings": 2 * temperatures + 3 * (60 - temperatures),
            "first_reading": "2016-01-23 12:10:00",
            "first_temperature": "2016-05-03 09:40:00",
            "last_reading": "2016-05-03 09:50:00",
            "steps_of_ten_minutes": 0.66667,
        }
    ]


def test_the_turbines_held_unread_are_the_first_of_the_seeds_order() -> None:
    kelmarsh = GreenbyteCorpusReader.KELMARSH
    penmanshiel = GreenbyteCorpusReader.PENMANSHIEL

    # Pinned: another framing of the seeded order is another draw of what stays unread.
    assert held_out(kelmarsh.name, kelmarsh.turbines, 2, seed=1) == ["2", "6"]
    assert held_out(penmanshiel.name, penmanshiel.turbines, 4, seed=1) == ["05", "09", "11", "12"]
    assert held_out(kelmarsh.name, kelmarsh.turbines, 2, seed=2) != ["2", "6"]


def test_excerpts_share_real_time_by_their_wind_or_by_their_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wind_fleet_audit, "LEAST_SHARED", 10)
    rng = np.random.default_rng(3)
    stamps = pd.date_range("2022-01-01", periods=300, freq="10min")
    one_year = rng.normal(size=300)
    another_year = rng.normal(size=300)
    wind = {
        # Two turbines in one real year, a third excerpt of the first turbine with the same rows.
        10: pd.Series(one_year, index=stamps),
        11: pd.Series(one_year + 0.05 * rng.normal(size=300), index=stamps),
        12: pd.Series(one_year, index=stamps),
        # The first turbine again in another year laid on the same stamps, and a turbine beside.
        20: pd.Series(another_year, index=stamps),
        21: pd.Series(another_year + 0.05 * rng.normal(size=300), index=stamps),
    }
    turbines = {10: 1, 11: 2, 12: 1, 20: 1, 21: 3}

    shared = shared_time_groups(wind, turbines)

    assert shared.groups == [[10, 11, 12], [20, 21]]
    # Excerpts 10 and 12 hold the same rows; 20 holds other rows than either at equal stamps.
    assert (shared.same_rows, shared.other_rows) == (1, 2)
    assert shared.least_within == pytest.approx(1.0, abs=0.01)
    assert shared.most_between == pytest.approx(0.0, abs=0.2)
