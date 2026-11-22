"""The model of normal behaviour over an excerpt: what it is fitted on and what it answers."""

from collections.abc import Callable

import numpy as np
import pytest

from scripts.care_excerpts import Event, Hours, Sensors
from scripts.care_normal_behaviour_scores import (
    FAMILY,
    FLEET,
    FOLDS,
    HOURS_PER_WEEK,
    LOADS,
    Hourly,
    answers_of,
    behaviour_scores,
    deviations,
    explained,
    fleet_answers,
    folds_of,
    hourly_mean,
    loads_before,
    others_of,
    reduced,
    ridge_fit,
)
from tests.scripts.test_care_excerpts import excerpt

HOURS, FITTED = 24 * 7 * 14, 24 * 7 * 12


def hourly(seed: int = 1, *, offset_from: int | None = None) -> Hourly:
    """A temperature that follows the power, and one that follows nothing."""
    draws = np.random.default_rng(seed)
    power = draws.random(HOURS)
    features = np.column_stack([power, draws.random(HOURS)])
    followed = 30.0 + 40.0 * power + 0.5 * draws.standard_normal(HOURS)
    if offset_from is not None:
        followed[offset_from:] += 5.0
    targets = np.column_stack([followed, 10.0 + draws.standard_normal(HOURS)])
    return Hourly(features=features, targets=targets, fitted=FITTED)


class Remembering:
    """A model standing in: answers the fitted targets' mean and remembers what it was fitted on."""

    def __init__(self) -> None:
        self.fitted_on: list[np.ndarray] = []

    def __call__(
        self, rows: np.ndarray, targets: np.ndarray, seed: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        self.fitted_on.append(rows.copy())
        level = float(targets.mean())
        return lambda new: np.full(len(new), level)


def test_the_power_before_an_hour_leaves_the_hour_itself_out() -> None:
    power = np.arange(30.0)

    loads = loads_before(power)

    assert loads.shape == (30, len(LOADS))
    assert loads[10, 0] == np.mean([7.0, 8.0, 9.0])
    assert loads[29, 2] == np.mean(np.arange(5.0, 29.0))


def test_a_load_with_fewer_than_half_its_hours_is_missing() -> None:
    power = np.arange(30.0)
    power[4:9] = np.nan

    loads = loads_before(power)

    assert np.isnan(loads[0]).all()
    assert loads[2, 0] == 0.5
    assert np.isnan(loads[8, 0])
    # One of the three hours before hour 10 holds a reading, two of those before hour 11.
    assert np.isnan(loads[10, 0])
    assert loads[11, 0] == 9.5
    assert np.isnan(loads[11, 2])


def test_the_same_hours_give_the_same_load_wherever_an_excerpt_starts() -> None:
    power = np.random.default_rng(6).random(400) * 2000.0

    whole, later = loads_before(power), loads_before(power[137:])

    assert np.array_equal(later[max(LOADS) :], whole[137 + max(LOADS) :])


def test_a_stopped_turbines_rows_count_in_the_hours_mean_power() -> None:
    values = np.full((18, 3), 7.0)
    values[:, 0] = 0.0
    values[6:12, 0] = [0.0, 0.0, 0.0, 6.0, 6.0, np.nan]
    status = np.zeros(18, dtype=np.int64)
    status[6:9] = 4
    read = excerpt(values, fitted_rows=12, event_rows=(12, 17), status=status)

    means = hourly_mean(read.values[:, 0], Hours.of(read))

    assert means.tolist() == [0.0, 2.4, 0.0]


def test_the_weeks_are_dealt_to_the_folds_in_turn() -> None:
    folds = folds_of(HOURS_PER_WEEK * 13)

    assert set(folds[:HOURS_PER_WEEK]) == {0}
    assert folds[HOURS_PER_WEEK * 7] == 1
    assert sorted(set(folds)) == list(range(FOLDS))


def test_a_fitted_hour_is_answered_by_a_model_that_never_read_it() -> None:
    read = hourly()
    model = Remembering()

    answers = answers_of(read, 0, model, seed=1)

    assert len(model.fitted_on) == FOLDS + 1
    # The scored window is answered by the model of the whole fitted year.
    assert len(model.fitted_on[-1]) == FITTED
    assert all(len(rows) < FITTED for rows in model.fitted_on[:FOLDS])
    folds = folds_of(HOURS)
    for fold, rows in enumerate(model.fitted_on[:FOLDS]):
        held = read.features[:FITTED][folds[:FITTED] == fold]
        assert not (rows[:, None, :] == held[None, :50, :]).all(axis=2).any()
    assert not np.isnan(answers).any()


def test_no_hour_of_the_scored_window_is_fitted_on() -> None:
    read = hourly()
    model = Remembering()

    answers_of(read, 0, model, seed=1)

    scored = read.features[FITTED:]
    for rows in model.fitted_on:
        assert len(rows) <= FITTED
        assert not (rows[:, None, 0] == scored[None, :40, 0]).any()


def test_an_hour_without_its_conditions_has_no_answer_and_is_not_fitted_on() -> None:
    read = hourly()
    read.features[100, 0] = np.nan
    read.targets[200, 0] = np.nan
    model = Remembering()

    answers = answers_of(read, 0, model, seed=1)

    assert np.isnan(answers[100])
    assert not np.isnan(answers[200])
    assert all(np.isfinite(rows).all() for rows in model.fitted_on)


def test_the_farms_model_is_fitted_on_the_other_turbines_alone_each_hour_once() -> None:
    own, other = hourly(1), hourly(2)
    model = Remembering()

    answers = fleet_answers(own, [other, other], 0, model, seed=1)

    (rows,) = model.fitted_on
    assert len(rows) == FITTED
    assert not (rows[:, None, 0] == own.features[None, :40, 0]).any()
    assert not np.isnan(answers).any()


def test_two_excerpts_of_a_turbine_starting_apart_are_fitted_on_hour_by_hour_once() -> None:
    other = hourly(2)
    apart = Hourly(features=other.features[300:], targets=other.targets[300:], fitted=FITTED - 300)
    model = Remembering()

    fleet_answers(hourly(1), [other, apart], 0, model, seed=1)

    (rows,) = model.fitted_on
    assert len(rows) == FITTED
    # In the order the hours were read, not sorted by their values.
    assert np.array_equal(rows, other.features[:FITTED])


def test_a_farms_model_reads_the_other_turbines_excerpts_alone() -> None:
    own = Event("C", 1, 12, True, 0, 1)
    every = [
        own,
        Event("C", 2, 12, False, 0, 1),
        Event("C", 3, 15, True, 0, 1),
        Event("B", 4, 7, True, 0, 1),
    ]

    assert [other.event for other in others_of(own, every)] == [3]


def test_nothing_of_the_scored_window_enters_a_model_or_its_residuals_scale() -> None:
    plain, other = hourly(), hourly(3)
    later = Hourly(features=plain.features.copy(), targets=plain.targets.copy(), fitted=FITTED)
    later.features[FITTED:] += 0.3
    later.targets[FITTED:] += 25.0

    scores, _ = behaviour_scores(plain, [other], ridge_fit, seed=1)
    moved, _ = behaviour_scores(later, [other], ridge_fit, seed=1)

    for name, series in scores.items():
        assert np.array_equal(moved[name][:FITTED], series[:FITTED], equal_nan=True), name


def test_a_ridge_answers_a_temperature_that_follows_the_power() -> None:
    read = hourly()

    answers = answers_of(read, 0, ridge_fit, seed=1)

    assert explained(answers, read.targets[:, 0], FITTED) > 0.95
    assert explained(answers_of(read, 1, ridge_fit, seed=1), read.targets[:, 1], FITTED) < 0.05


def test_a_deviation_is_measured_in_the_fitted_hours_residuals() -> None:
    values = np.r_[np.tile([9.0, 11.0], 50), 20.0]
    answers = np.full(101, 10.0)

    far = deviations(answers, values, fitted=100)

    assert far[:100].tolist() == pytest.approx([1.0 / 1.4826] * 100)
    assert far[100] == pytest.approx(10.0 / 1.4826)


def test_residuals_that_mostly_agree_are_measured_in_standard_deviations() -> None:
    values = np.r_[np.full(90, 10.0), np.full(10, 12.0), 14.0]

    far = deviations(np.full(101, 10.0), values, fitted=100)

    assert far[100] == pytest.approx(4.0 / np.r_[np.zeros(90), np.full(10, 2.0)].std())


def test_an_hour_no_temperature_answers_has_no_score() -> None:
    far = np.array([[1.0, 3.0], [np.nan, 2.0], [np.nan, np.nan]])

    scores = reduced(far)

    assert scores["max"][:2].tolist() == [3.0, 2.0]
    assert scores["mean"][:2].tolist() == [2.0, 2.0]
    assert np.isnan(scores["max"][2])
    assert np.isnan(scores["mean"][2])


def test_a_temperature_that_leaves_its_answer_scores_high_in_the_scored_window() -> None:
    read = hourly(offset_from=FITTED + 24)

    scores, shares = behaviour_scores(read, [hourly(3)], ridge_fit, seed=1)

    assert set(scores) == {
        f"{family}/{name}" for family in (FAMILY, FLEET) for name in ("max", "mean")
    }
    assert shares[0] > 0.95
    for family in (FAMILY, FLEET):
        series = scores[f"{family}/max"]
        assert np.median(series[FITTED + 24 :]) > 5 * np.median(series[:FITTED])


def test_an_excerpt_alone_in_its_farm_has_no_farms_model() -> None:
    scores, _ = behaviour_scores(hourly(), [], ridge_fit, seed=1)

    assert set(scores) == {f"{FAMILY}/max", f"{FAMILY}/mean"}


def test_an_excerpt_is_read_by_its_conditions_its_loads_and_its_temperatures() -> None:
    draws = np.random.default_rng(5)
    rows = 6 * 40
    values = 10.0 + draws.random((rows, 4))
    read = excerpt(values, fitted_rows=6 * 30, event_rows=(6 * 32, rows - 1))
    sensors = Sensors(
        channels=read.columns, temperatures=read.columns[2:], conditions=read.columns[:2]
    )

    laid = Hourly.of(read, Hours.of(read), sensors)

    assert laid.features.shape == (40, 2 + len(LOADS))
    assert laid.targets.shape == (40, 2)
    assert laid.fitted == 30
    assert laid.features[5, 1] == pytest.approx(values[30:36, 1].mean())
    # The second condition is the power, whose mean over the three hours before is the load.
    assert laid.features[5, 2] == pytest.approx(values[12:30, 1].mean())
    assert laid.targets[5, 0] == pytest.approx(values[30:36, 2].mean())
