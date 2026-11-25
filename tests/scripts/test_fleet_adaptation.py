"""What the forecaster is fine-tuned on: series before a cut, nothing of the turbines it answers."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.fleet_adaptation import (
    LEAST_READ,
    SPAN,
    FitRecord,
    folds,
    mission_series,
    refused,
    spans,
    training_series,
    turbine_series,
)
from scripts.fleet_hours import CHANNELS, CONDITIONS, TARGETS, Turbine, axis_of, hour_of

CUT = date(2020, 1, 1)
END = hour_of(CUT) + 24 * 400


def turbine(name: str, level: float = 1.0, first: int = 0) -> Turbine:
    """Read from hour ``first`` to a year past the cut; each hour's value is its index."""
    values = np.full((END, len(CHANNELS)), np.nan)
    values[first:] = level + np.arange(first, END, dtype=np.float64)[:, None]
    return Turbine(name, values)


def record(**changed: object) -> FitRecord:
    fields: dict[str, object] = {
        "name": "penmanshiel-fold0",
        "arm": "adapted_fleet",
        "farm": "penmanshiel",
        "answers": ("01", "07"),
        "turbines": ("penmanshiel/02", "penmanshiel/04"),
        "missions": (),
        "share": 0.0,
        "turbine_series": 100,
        "mission_series": 0,
        "until": "2020-01-01",
        "model": "amazon/chronos-2",
        "mode": "lora",
        "rank": 8,
        "steps": 2000,
        "learning_rate": 1e-5,
        "schedule": "linear",
        "context": 2048,
        "batch": 256,
        "save_every": 0,
        "seed": 1,
    }
    fields.update(changed)
    return FitRecord(**fields)  # type: ignore[arg-type]


def test_folds_deal_the_sorted_names_in_turn_and_refuse_more_folds_than_names() -> None:
    names = ["07", "01", "15", "02", "04"]

    assert folds(names, 2) == [("01", "04", "15"), ("02", "07")]
    assert folds(names, 5) == [("01",), ("02",), ("04",), ("07",), ("15",)]
    with pytest.raises(ValueError, match="cannot be dealt"):
        folds(names, 6)
    with pytest.raises(ValueError, match="cannot be dealt"):
        folds(names, 0)


def test_a_turbines_spans_end_before_the_cut_and_start_at_its_first_reading() -> None:
    first = hour_of(CUT) - 2 * (SPAN + 1) - 100
    series = turbine_series(turbine("01", first=first), hour_of(CUT))

    assert [s["target"].shape for s in series] == [
        (len(TARGETS), SPAN + 1),
        (len(TARGETS), SPAN + 1),
        (len(TARGETS), 100),
    ]
    assert series[0]["target"][0, 0] == 1.0 + first
    assert series[-1]["target"][0, -1] == 1.0 + hour_of(CUT) - 1
    assert set(series[0]["past_covariates"]) == {f"s{i}" for i in CONDITIONS}
    assert series[1]["past_covariates"][f"s{CONDITIONS[1]}"][0] == 1.0 + first + SPAN + 1
    assert series[0]["future_covariates"] == {f"s{i}": None for i in CONDITIONS}
    assert turbine_series(turbine("01", first=hour_of(CUT) - LEAST_READ + 1), hour_of(CUT)) == []


def test_a_span_reading_too_few_hours_is_left_out() -> None:
    read = np.zeros(3 * (SPAN + 1), dtype=bool)
    read[5 : 5 + LEAST_READ] = True
    read[5 + SPAN + 1 : 5 + SPAN + LEAST_READ] = True
    read[5 + 2 * (SPAN + 1) :] = True

    assert spans(read) == [slice(5, 5 + SPAN + 1), slice(5 + 2 * (SPAN + 1), 3 * (SPAN + 1))]
    assert spans(np.zeros(10, dtype=bool)) == []


def test_the_series_read_nothing_of_the_turbines_left_out_nor_from_the_cut() -> None:
    farm = [turbine("01"), turbine("02", 5.0), turbine("04", 9.0)]
    trained = training_series({"penmanshiel": farm}, left_out={"02"}, until=CUT)

    changed = [Turbine(t.name, t.values.copy()) for t in farm]
    changed[1].values[:] = -1.0
    for t in changed:
        t.values[hour_of(CUT) :] = -1.0
    again = training_series({"penmanshiel": changed}, left_out={"02"}, until=CUT)

    assert trained.turbines == ("penmanshiel/01", "penmanshiel/04")
    assert (
        trained.turbine_series
        == len(trained.series)
        == 2 * len(turbine_series(farm[0], hour_of(CUT)))
    )
    assert trained.mission_series == 0
    assert len(again.series) == len(trained.series)
    for a, b in zip(trained.series, again.series, strict=True):
        assert np.array_equal(a["target"], b["target"])


def test_the_missions_take_their_share_of_the_series_in_groups_of_at_most_seventeen() -> None:
    farm = [turbine(name, first=hour_of(CUT) - 3 * (SPAN + 1)) for name in ("1", "3")]
    missions = {"M1": np.ones((10 * (SPAN + 1), 20)), "M2": np.ones((4 * (SPAN + 1), 17))}

    trained = training_series(
        {"kelmarsh": farm}, left_out=(), until=CUT, missions=missions, share=0.5, seed=3
    )
    pool = mission_series(missions["M1"])

    assert trained.turbine_series == 6
    assert trained.mission_series == 6
    assert len(trained.series) == 12
    assert all("past_covariates" not in s for s in trained.series[6:])
    assert sorted({s["target"].shape[0] for s in pool}) == [3, 17]
    assert len(pool) == 20
    with pytest.raises(ValueError, match="hold 24 series, not the 54 wanted"):
        training_series({"kelmarsh": farm}, left_out=(), until=CUT, missions=missions, share=0.9)
    with pytest.raises(ValueError, match="cannot take a share"):
        training_series({"kelmarsh": farm}, left_out=(), until=CUT, missions=missions, share=1.0)


def test_a_turbine_held_unread_a_fold_not_loaded_or_a_cut_past_the_years_is_refused() -> None:
    with pytest.raises(ValueError, match="held unread"):
        training_series({"penmanshiel": [turbine("05")]}, left_out=(), until=CUT)
    with pytest.raises(ValueError, match="left out but not loaded"):
        training_series({"penmanshiel": [turbine("01")]}, left_out={"09x"}, until=CUT)
    with pytest.raises(ValueError, match="may read nothing from 2023"):
        training_series({"penmanshiel": [turbine("01")]}, left_out=(), until=date(2023, 1, 2))


def test_a_fit_refuses_cells_of_turbines_it_read_does_not_answer_or_that_start_before_it() -> None:
    fit = record(answers=("01", "02"))
    after, before = date(2020, 3, 1), date(2019, 12, 1)

    assert refused(fit, [(10, axis_of("01", after), 0)]) == []
    reasons = refused(
        fit,
        [(1, axis_of("02", after), 0), (2, axis_of("07", after), 0), (3, axis_of("01", before), 0)],
    )
    assert any("read turbine 02" in r for r in reasons)
    assert any("does not answer turbine 07" in r for r in reasons)
    assert any("starts before" in r for r in reasons)


def test_a_fits_record_reads_back_as_written(tmp_path: Path) -> None:
    fit = record(missions=("ESA-Mission1",), share=0.5, mission_series=100)
    fit.write(tmp_path)

    assert FitRecord.read(tmp_path) == fit
