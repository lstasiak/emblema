import numpy as np
import pytest

from emblema.evaluation.adapters.detection.standardisation import Standardisation

ROWS = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 2.0], [5.0, 5.0, 4.0]])
# The spreads are 1.633, 0 and 1.0, so the typical spread is 1.0 and the floor a tenth of it.
FLOOR = 0.1


def test_a_feature_without_spread_is_read_over_a_tenth_of_the_typical_spread() -> None:
    standardisation = Standardisation.fitted_on(ROWS)

    assert standardisation.scale[1] == pytest.approx(FLOOR)
    assert standardisation.of(np.array([[3.0, 7.0, 3.0]]))[0, 1] == pytest.approx(2.0 / FLOOR)


def test_a_feature_with_spread_above_the_floor_keeps_its_own() -> None:
    standardisation = Standardisation.fitted_on(ROWS)

    assert standardisation.scale[0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert standardisation.scale[2] == pytest.approx(1.0)


def test_rows_without_any_spread_are_read_over_one() -> None:
    standardisation = Standardisation.fitted_on(np.array([[5.0, 2.0], [5.0, 2.0]]))

    assert standardisation.scale.tolist() == [1.0, 1.0]
    assert standardisation.of(np.array([[7.0, 2.0]]))[0].tolist() == [2.0, 0.0]


def test_one_fitting_row_moving_a_constant_feature_leaves_its_scale_at_the_floor() -> None:
    # A feature constant on a thousand rows, then the same rows with one moved by 0.01: the
    # spread goes from nought to 3e-4, and the feature is read over the floor both times.
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.normal(size=1000), rng.normal(size=1000), np.full(1000, 3.0)])
    moved = rows.copy()
    moved[0, 2] += 0.01

    before = Standardisation.fitted_on(rows).scale[2]
    after = Standardisation.fitted_on(moved).scale[2]

    assert before == pytest.approx(after, rel=1e-3)
    assert before == pytest.approx(0.1 * np.median(np.std(rows, axis=0)))


def test_where_more_than_half_the_features_have_no_spread_the_floor_follows_those_that_do() -> None:
    standardisation = Standardisation.fitted_on(
        np.array([[1.0, 5.0, 5.0], [3.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    )

    assert standardisation.scale.tolist() == pytest.approx(
        [np.std([1.0, 3.0, 5.0]), 0.1 * np.std([1.0, 3.0, 5.0]), 0.1 * np.std([1.0, 3.0, 5.0])]
    )


def test_a_missing_statistic_is_read_at_the_fitting_mean() -> None:
    standardisation = Standardisation.fitted_on(ROWS)

    assert standardisation.centre[2] == 3.0
    assert standardisation.of(np.array([[3.0, 5.0, np.nan]]))[0, 2] == 0.0
    assert standardisation.of(ROWS)[1].tolist() == [0.0, 0.0, -1.0]
