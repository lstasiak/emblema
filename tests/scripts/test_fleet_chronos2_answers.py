"""The published forecaster on a new turbine: what each rung is handed and what comes back."""

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.chronos2_ladder import CONTEXT
from scripts.fleet_chronos2_answers import (
    CONCURRENT,
    NBM,
    NBM_CURRENTS,
    NBM_OWN,
    NEIGHBOURS,
    OWN,
    RUNGS,
    answered,
    nbm_input,
    neighbours_mean,
    rung_input,
)
from scripts.fleet_hours import CHANNELS, CONDITIONS, TARGETS, WINDOW, Turbine, axis_of, hour_of

START = date(2021, 3, 1)
BEGIN = hour_of(START)


def turbine(name: str, level: float, hours: int = BEGIN + 400) -> Turbine:
    values = np.full((hours, len(CHANNELS)), level)
    return Turbine(name, values)


def test_the_neighbours_mean_leaves_out_the_new_turbine_and_hours_none_read() -> None:
    a, b, c = turbine("1", 1.0, 10), turbine("2", 3.0, 10), turbine("3", 5.0, 10)
    b.values[4, 0] = np.nan
    c.values[4, 0] = np.nan

    mean = neighbours_mean([a, b, c], "1")

    assert mean.shape == (10, len(TARGETS))
    assert mean[0, 0] == 4.0
    assert np.isnan(mean[4, 0])
    assert mean[4, 1] == 4.0


def test_each_rung_hands_its_hours_and_nothing_before_the_start_or_from_the_hour() -> None:
    new = turbine("1", 2.0).before(BEGIN)
    neighbours = np.arange(float(BEGIN + 400))[:, None] * np.ones((1, len(TARGETS)))
    window = BEGIN + 60

    plain = rung_input("chronos2_47", window, BEGIN, new.values, neighbours, cap=8192)
    own = rung_input(OWN, window, BEGIN, new.values, neighbours, cap=8192)
    capped = rung_input(OWN, window, BEGIN, new.values, neighbours, cap=30)
    known = rung_input(NEIGHBOURS, window, BEGIN, new.values, neighbours, cap=8192)
    ceiling = rung_input(CONCURRENT, window, BEGIN, new.values, neighbours, cap=8192)

    assert plain.shape == (len(WINDOW), CONTEXT)
    assert own.shape == (len(WINDOW), 60)
    assert np.isfinite(own).all()
    assert capped.shape == (len(WINDOW), 30)
    assert set(known) == {"target", "past_covariates"}
    assert known["past_covariates"]["n0"].tolist() == list(range(window - CONTEXT, window))
    assert ceiling["future_covariates"]["n16"].tolist() == [float(window)]
    assert len(ceiling["past_covariates"]) == len(TARGETS)


def echo(handed: Sequence[Any]) -> list[NDArray[np.float32]]:
    """A forecaster standing in: each variate's last hour handed, as every quantile."""
    out = []
    for item in handed:
        target = item["target"] if isinstance(item, dict) else item
        last = target[:, -1].astype(np.float32)
        out.append(np.repeat(last[:, None, None], 3, axis=1))
    return out


def test_every_rung_answers_every_scored_cell_and_only_those() -> None:
    turbines = [turbine("1", 1.0), turbine("2", 2.0)]
    axis = axis_of("1", START)
    cells = [(BEGIN + 50, axis, 0), (BEGIN + 50, axis, 16), (BEGIN + 70, axis, 3)]

    out = answered(cells, turbines, echo, median=1, cap=8192, batch=2)

    assert set(out) == set(RUNGS)
    for rung in RUNGS:
        assert out[rung] == dict.fromkeys(cells, 1.0)


def test_a_cell_of_a_turbine_not_loaded_or_an_answer_of_other_variates_is_refused() -> None:
    turbines = [turbine("1", 1.0), turbine("2", 2.0)]
    cells = [(BEGIN + 50, axis_of("1", START), 0)]

    def short(handed: Sequence[Any]) -> list[NDArray[np.float32]]:
        return [np.zeros((3, 3, 1), dtype=np.float32) for _ in handed]

    with pytest.raises(ValueError, match="not loaded"):
        answered([(BEGIN + 50, axis_of("9", START), 0)], turbines, echo, median=1, cap=64)
    with pytest.raises(ValueError, match="variates"):
        answered(cells, turbines, short, median=1, cap=64)


def test_the_forecaster_never_sees_the_new_turbine_before_its_start_nor_in_its_neighbours() -> None:
    new = turbine("1", 1.0)
    new.values[:BEGIN] = 100.0
    turbines = [new, turbine("2", 3.0), turbine("3", 5.0)]
    handed: list[Any] = []

    def recorded(batch: Sequence[Any]) -> list[NDArray[np.float32]]:
        handed.extend(batch)
        return echo(batch)

    answered([(BEGIN + 30, axis_of("1", START), 0)], turbines, recorded, median=1, cap=8192)

    targets = [item["target"] if isinstance(item, dict) else item for item in handed]
    assert all(not (np.asarray(t) == 100.0).any() for t in targets)
    covariates = [
        v
        for item in handed
        if isinstance(item, dict)
        for k, v in item["past_covariates"].items()
        if k.startswith("n")
    ]
    conditions = [
        v
        for item in handed
        if isinstance(item, dict)
        for k, v in item["past_covariates"].items()
        if k.startswith("s")
    ]
    assert conditions
    assert all(not (np.asarray(v) == 100.0).any() for v in conditions)
    assert covariates
    assert all((np.asarray(v) == 4.0).all() for v in covariates)


def test_a_normal_behaviour_rung_hands_the_conditions_before_and_in_the_hour_or_nothing() -> None:
    new = turbine("1", 2.0).before(BEGIN)
    new.values[:, CONDITIONS[2]] = np.arange(float(len(new.values)))
    window = BEGIN + 60

    handed = nbm_input(NBM, window, BEGIN, new.values, cap=8192)
    wider = nbm_input(NBM_CURRENTS, window, BEGIN, new.values, cap=8192)
    longer = nbm_input(NBM_OWN, window, BEGIN, new.values, cap=8192)
    capped = nbm_input(NBM_OWN, window, BEGIN, new.values, cap=30)
    new.values[window, CONDITIONS[0]] = np.nan
    missing = nbm_input(NBM, window, BEGIN, new.values, cap=8192)

    assert handed is not None
    assert wider is not None
    assert longer is not None
    assert capped is not None
    assert longer["target"].shape == (len(TARGETS), 60)
    assert np.isfinite(longer["target"]).all()
    assert longer["past_covariates"][f"s{CONDITIONS[2]}"].tolist() == list(range(BEGIN, window))
    assert capped["target"].shape == (len(TARGETS), 30)
    assert handed["target"].shape == (len(TARGETS), CONTEXT)
    assert set(handed["past_covariates"]) == {f"s{i}" for i in CONDITIONS}
    key = f"s{CONDITIONS[2]}"
    assert handed["past_covariates"][key].tolist() == list(range(window - CONTEXT, window))
    assert handed["future_covariates"][key].tolist() == [float(window)]
    assert len(wider["future_covariates"]) == len(CONDITIONS) + 3
    assert missing is None


def test_the_normal_behaviour_rungs_answer_the_temperatures_where_the_hour_is_known() -> None:
    turbines = [turbine("1", 1.0), turbine("2", 2.0)]
    axis = axis_of("1", START)
    gap = BEGIN + 70
    turbines[0].values[gap, CONDITIONS[1]] = np.nan
    cells = [(BEGIN + 50, axis, 0), (gap, axis, 3)]

    out = answered(cells, turbines, echo, median=1, cap=8192)

    for rung in (NBM, NBM_CURRENTS, NBM_OWN):
        assert out[rung] == {(BEGIN + 50, axis, 0): 1.0}


def test_only_the_rungs_asked_for_are_answered() -> None:
    turbines = [turbine("1", 1.0), turbine("2", 2.0)]
    cells = [(BEGIN + 50, axis_of("1", START), 0)]

    out = answered(cells, turbines, echo, median=1, cap=8192, rungs=(NBM, NBM_OWN))

    assert set(out) == {NBM, NBM_OWN}
    assert out[NBM] == out[NBM_OWN] == {cells[0]: 1.0}
