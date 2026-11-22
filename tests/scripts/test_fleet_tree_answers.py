"""The farm's trees for a new turbine: fitted on the others before the start's year."""

from collections.abc import Callable
from datetime import date

import numpy as np
import pytest

from scripts.fleet_hours import CHANNELS, CONDITIONS, READ_UNTIL, Turbine, axis_of, hour_of
from scripts.fleet_tree_answers import MOST_ROWS, TREES, answered, drawn

START = date(2021, 3, 1)
YEAR = hour_of(date(2021, 1, 1))


def farm(seed: int = 5) -> list[Turbine]:
    """Three turbines read from 100 days before the start's year to 60 days after the start."""
    rng = np.random.default_rng(seed)
    begin, end = YEAR - 100 * 24, hour_of(START) + 60 * 24
    turbines = []
    for name in ("1", "2", "3"):
        values = np.full((hour_of(READ_UNTIL), len(CHANNELS)), np.nan)
        values[begin:end] = np.cumsum(rng.normal(size=(end - begin, len(CHANNELS))), axis=0) * 0.1
        turbines.append(Turbine(name, values))
    return turbines


def mean_of_targets(
    rows: np.ndarray, targets: np.ndarray, seed: int
) -> Callable[[np.ndarray], np.ndarray]:
    """A model standing in: answers the fitted targets' mean plus each row's last feature."""
    level = float(targets.mean()) + 1e-3 * float(rows.sum())

    def answer(new: np.ndarray) -> np.ndarray:
        return level + new[:, -1]

    return answer


def cells(hours: range) -> list[tuple[int, str, int]]:
    return [(h, axis_of("1", START), c) for h in hours for c in (0, 10)]


def test_the_trees_read_nothing_of_the_new_turbine_nor_of_the_start_year() -> None:
    turbines = farm()
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 260, 7))
    first = answered(turbines, keys, mean_of_targets, seed=1)

    changed = [Turbine(t.name, t.values.copy()) for t in turbines]
    changed[0].values[: hour_of(START)] += 50.0
    for other in changed[1:]:
        other.values[YEAR:] -= 9.0
    again = answered(changed, keys, mean_of_targets, seed=1)

    for tree in TREES:
        assert first[tree]
        assert again[tree] == pytest.approx(first[tree])


def test_the_normal_behaviour_trees_answer_only_where_the_hours_conditions_read() -> None:
    turbines = farm()
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 230))
    gap = keys[4][0]
    turbines[0].values[gap, CONDITIONS[3]] = np.nan

    out = answered(turbines, keys, mean_of_targets, seed=1)

    assert {k[0] for k in out["fleet_nbm_trees"]} == {k[0] for k in keys} - {gap}
    assert {k[0] for k in out["fleet_trees"]} == {k[0] for k in keys}


def test_a_fit_reads_at_most_its_rows_drawn_by_seed_in_their_order() -> None:
    rows = np.arange(MOST_ROWS + 50, dtype=np.float64)[:, None]
    targets = rows[:, 0] * 2

    kept, aims = drawn(rows, targets, seed=3)

    assert len(kept) == MOST_ROWS
    assert (np.diff(kept[:, 0]) > 0).all()
    assert (aims == kept[:, 0] * 2).all()
    assert (drawn(rows, targets, seed=3)[0] == kept).all()
    assert len(drawn(rows[:10], targets[:10], seed=3)[0]) == 10


def test_a_cell_of_a_turbine_not_loaded_is_refused() -> None:
    with pytest.raises(ValueError, match="not loaded"):
        answered(farm(), [(hour_of(START) + 200, axis_of("9", START), 0)], mean_of_targets, seed=1)
