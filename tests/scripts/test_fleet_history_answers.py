"""The trees' answers corrected by a new turbine's own hours: what each correction may read."""

from collections.abc import Callable

import numpy as np
import pytest

from scripts.cold_start_cells import CONTEXT
from scripts.fleet_cold_start_cells import FLEET_NBM, FLEET_ROWS, latest
from scripts.fleet_history_answers import (
    LEAST_ROWS,
    PRIOR_HOURS,
    SHRUNK_FIRST,
    SUFFIXES,
    History,
    boosted,
    corrected,
    history_of,
    offsets,
    ridged,
)
from scripts.fleet_hours import CHANNELS, CONDITIONS

KIND = next(r for r in FLEET_ROWS if r.name == FLEET_NBM)
START = 500
CHANNEL = 3


def hours_of(*hours: int) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.asarray(hours, dtype=np.int64)


def last_hour(rows: np.ndarray) -> np.ndarray:
    """A model standing in for the trees: the channel's hour before."""
    out: np.ndarray = rows[:, CONTEXT - 1].copy()
    return out


def turbine(hours: int = 400, seed: int = 2) -> np.ndarray:
    """A turbine's values, unknown before the start, a smooth walk after it."""
    rng = np.random.default_rng(seed)
    values = np.full((START + hours, len(CHANNELS)), np.nan)
    values[START:] = np.cumsum(rng.normal(size=(hours, len(CHANNELS))), axis=0) * 0.05
    return values


def history(hours: np.ndarray, residuals: np.ndarray, conditions: np.ndarray) -> History:
    rows = np.column_stack([hours.astype(np.float64), residuals])
    return History(hours.astype(np.int64), rows, np.zeros(len(hours)), conditions, residuals)


def test_the_history_holds_the_own_hours_the_trees_answer_before_the_end() -> None:
    values = turbine()
    values[START + 100, CONDITIONS[2]] = np.nan
    end = START + 300

    own = history_of(values, latest(values), KIND, CHANNEL, START, end, last_hour)

    # Scored from the hour half a context has been read; not where a condition is missing.
    assert own.hours[0] == START + CONTEXT // 2 + 1
    assert own.hours[-1] == end - 1
    assert START + 100 not in own.hours
    assert own.residuals == pytest.approx(
        values[own.hours, CHANNEL] - values[own.hours - 1, CHANNEL]
    )
    assert own.conditions == pytest.approx(values[own.hours][:, list(CONDITIONS)])
    nothing = history_of(values, latest(values), KIND, CHANNEL, START, START + 5, last_hour)
    assert len(nothing.hours) == 0
    assert nothing.conditions.shape == (0, len(CONDITIONS))


def test_the_offset_is_the_mean_before_the_hour_shrunk_by_a_day() -> None:
    hours = np.arange(600, 700)
    own = history(hours, np.full(100, 2.0), np.zeros((100, 1)))

    given = offsets(own, hours_of(600, 624, 700, 900))

    whole = 2.0 * 100 / (100 + PRIOR_HOURS)
    assert given == pytest.approx([0.0, 2.0 * 24 / (24 + PRIOR_HOURS), whole, whole])


def test_the_ridge_learns_a_turbines_own_response_to_a_condition() -> None:
    rng = np.random.default_rng(4)
    hours = np.arange(600, 900)
    conditions = rng.normal(size=(300, 2))
    residuals = 0.3 + 0.5 * conditions[:, 0] + rng.normal(size=300) * 0.01
    own = history(hours, residuals, conditions)
    new = np.asarray([[2.0, 0.0], [-1.0, 3.0]])

    given = ridged(own, hours_of(900, 900), np.zeros(2), new)
    early = ridged(own, hours_of(600, 600), np.zeros(2), new)

    assert given == pytest.approx([1.3, -0.2], abs=0.05)
    assert early == pytest.approx([0.0, 0.0])


def test_the_ridge_is_the_one_fitted_hour_by_hour_on_the_hours_before() -> None:
    rng = np.random.default_rng(12)
    hours = np.sort(rng.choice(np.arange(600, 700), 60, replace=False))
    conditions = rng.normal(size=(60, 2))
    answers = rng.normal(size=60)
    residuals = 0.4 * conditions[:, 1] + rng.normal(size=60) * 0.3
    own = History(hours.astype(np.int64), np.zeros((60, 1)), answers, conditions, residuals)
    cells = hours_of(630, 655, 699, 700)
    new, given = rng.normal(size=(4, 2)), rng.normal(size=4)
    known = dict(zip(hours.tolist(), residuals.tolist(), strict=True))

    def row(hour: int, answer: float, signals: np.ndarray) -> np.ndarray:
        return np.asarray([1.0, answer, known.get(hour - 1, 0.0), *signals])

    def fitted(count: int, penalty: float) -> np.ndarray:
        if not count:
            return np.zeros(5)
        read = np.stack([row(int(hours[i]), answers[i], conditions[i]) for i in range(count)])
        solved: np.ndarray = np.linalg.solve(
            read.T @ read + penalty * np.eye(5), read.T @ residuals[:count]
        )
        return solved

    slow = []
    for cell, answer, signals in zip(cells.tolist(), given, new, strict=True):
        count = int((hours < cell).sum())
        missed = [
            sum(
                (residuals[i] - row(int(hours[i]), answers[i], conditions[i]) @ fitted(i, p)) ** 2
                for i in range(count)
            )
            for p in SHRUNK_FIRST
        ]
        penalty = SHRUNK_FIRST[int(np.argmin(missed))]
        slow.append(row(cell, answer, signals) @ fitted(count, penalty))

    assert ridged(own, cells, given, new) == pytest.approx(slow, rel=1e-6, abs=1e-9)


def test_hours_that_tell_no_penalty_apart_leave_the_trees_answer_nearly_as_it_is() -> None:
    one = history(hours_of(600), np.asarray([0.8]), np.ones((1, 2)))

    given = ridged(one, hours_of(601), np.ones(1), np.ones((1, 2)))

    # The least penalty would answer almost the whole 0.8 again.
    assert abs(float(given[0])) < 1e-3


def test_the_ridge_leaves_residuals_of_noise_nearly_uncorrected() -> None:
    rng = np.random.default_rng(6)
    hours = np.arange(600, 760)
    conditions = rng.normal(size=(160, 6))
    own = history(hours, rng.normal(size=160), conditions)
    new = rng.normal(size=(50, 6))

    given = ridged(own, np.full(50, 760), np.zeros(50), new)

    # An unshrunk fit of eight coefficients on 160 hours of noise would answer about 0.2.
    assert np.sqrt((given**2).mean()) < 0.1


def test_no_correction_reads_the_cells_hour_or_any_after_it() -> None:
    rng = np.random.default_rng(8)
    hours = np.arange(600, 800)
    conditions = rng.normal(size=(200, 2))
    residuals = rng.normal(size=200)
    own = history(hours, residuals, conditions)
    later = residuals.copy()
    later[150:] += 30.0
    changed = history(hours, later, conditions)
    cells = hours_of(700, 750)
    new = rng.normal(size=(2, 2))

    def small(rows: np.ndarray, targets: np.ndarray, seed: int) -> Callable[..., np.ndarray]:
        return lambda asked: np.full(len(asked), targets.mean())

    def all_of(given: History) -> dict[str, np.ndarray]:
        return corrected(
            given, cells, np.zeros((2, 2)), np.zeros(2), new, start=500, small=small, seed=1
        )

    first, again = all_of(own), all_of(changed)

    assert tuple(first) == SUFFIXES
    for suffix in SUFFIXES:
        assert again[suffix] == pytest.approx(first[suffix])
        assert first[suffix][0] != pytest.approx(0.0)
    earlier = residuals.copy()
    earlier[:40] += 30.0
    moved = all_of(history(hours, earlier, conditions))
    for suffix in SUFFIXES:
        assert moved[suffix][0] != pytest.approx(first[suffix][0])


def test_the_small_trees_are_fitted_on_the_hours_before_the_cells_day_once_enough() -> None:
    hours = np.arange(START + 24, START + 400)
    own = history(hours, np.ones(len(hours)), np.zeros((len(hours), 1)))
    fitted: list[int] = []

    def small(rows: np.ndarray, targets: np.ndarray, seed: int) -> Callable[..., np.ndarray]:
        fitted.append(int(rows[:, 0].max()))
        return lambda asked: np.full(len(asked), 7.0)

    cells = hours_of(START + 50, START + 5 * 24 + 3, START + 5 * 24 + 20, START + 9 * 24)
    given = boosted(own, cells, np.zeros((4, 2)), START, small, seed=1)

    # Day 2 has read one day of residuals, short of the least; days 5 and 9 are fitted once.
    assert LEAST_ROWS == 48
    assert given == pytest.approx([0.0, 7.0, 7.0, 7.0])
    assert fitted == [START + 5 * 24 - 1, START + 9 * 24 - 1]
