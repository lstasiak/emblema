"""The forecaster's answers corrected by a new turbine's own hours, as the trees' are."""

from collections.abc import Callable
from datetime import date

import numpy as np
import pytest

from scripts.cold_start_cells import CONTEXT
from scripts.fleet_forecaster_history import corrected_cells, own_keys
from scripts.fleet_history_answers import PRIOR_HOURS, SUFFIXES
from scripts.fleet_hours import CHANNELS, CONDITIONS, READ_UNTIL, Turbine, axis_of, hour_of

START = date(2021, 3, 1)
AXIS = axis_of("1", START)


def turbines(seed: int = 3) -> dict[str, Turbine]:
    rng = np.random.default_rng(seed)
    values = np.full((hour_of(READ_UNTIL), len(CHANNELS)), np.nan)
    begin = hour_of(START) - 30 * 24
    values[begin : begin + 90 * 24] = rng.normal(size=(90 * 24, len(CHANNELS)))
    return {"1": Turbine("1", values)}


def nought(rows: np.ndarray, targets: np.ndarray, seed: int) -> Callable[..., np.ndarray]:
    return lambda asked: np.zeros(len(asked))


def test_the_own_hours_run_from_the_start_to_before_the_last_cell_where_read() -> None:
    farm = turbines()
    start = hour_of(START)
    farm["1"].values[start + 100, CONDITIONS[2]] = np.nan
    farm["1"].values[start + 120, 4] = np.nan
    last = start + 14 * 24
    cells = [(start + 2 * 24, AXIS, 0), (last, AXIS, 4)]

    keys = own_keys(farm, cells)
    hours = {k[0] for k in keys}

    # A channel is scored once enough of the 47 hours before are read, all since the start.
    assert min(hours) > start
    assert max(hours) == last - 1
    assert start + 100 not in hours
    assert (start + 120, AXIS, 4) not in keys
    assert (start + 120, AXIS, 3) in keys
    assert {k[2] for k in keys} == set(range(17))
    assert keys == sorted(keys, key=lambda k: (k[1], k[0], k[2]))


def test_a_cells_correction_reads_the_forecasters_misses_before_it_alone() -> None:
    farm = turbines()
    start = hour_of(START)
    values = farm["1"].values
    cell = start + 10 * 24
    own_hours = range(start + CONTEXT, cell + 24)
    # The forecaster misses each own hour of channel 0 by 2 below its reading.
    own = {(h, AXIS, 0): float(values[h, 0] - 2.0) for h in own_hours}
    cells = {(cell, AXIS, 0): 0.5}

    first = corrected_cells(farm, cells, own, small=nought, seed=1)
    later = {k: (v - 50.0 if k[0] >= cell else v) for k, v in own.items()}
    again = corrected_cells(farm, cells, later, small=nought, seed=1)

    n = sum(1 for h in own_hours if h < cell)
    assert tuple(first) == SUFFIXES
    assert first["_offset"][(cell, AXIS, 0)] == pytest.approx(0.5 + 2.0 * n / (n + PRIOR_HOURS))
    # The ridge learns most of the miss; its penalty shrinks the rest towards no correction.
    assert 0.5 + 1.5 < first["_ridge"][(cell, AXIS, 0)] < 0.5 + 2.0
    assert first["_boosted"][(cell, AXIS, 0)] == pytest.approx(0.5)
    for suffix in SUFFIXES:
        assert again[suffix] == pytest.approx(first[suffix])


def test_a_cell_with_no_own_hour_answered_keeps_the_forecasters_answer() -> None:
    farm = turbines()
    cell = (hour_of(START) + 5 * 24, AXIS, 3)

    out = corrected_cells(farm, {cell: 0.7}, {}, small=nought, seed=1)

    for suffix in SUFFIXES:
        assert out[suffix][cell] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="not loaded"):
        corrected_cells(farm, {(cell[0], "9@2021-03-01", 3): 0.7}, {}, small=nought, seed=1)
