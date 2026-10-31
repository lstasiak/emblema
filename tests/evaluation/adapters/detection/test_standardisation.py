import numpy as np

from emblema.evaluation.adapters.detection.standardisation import Standardisation

ROWS = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 2.0], [5.0, 5.0, 4.0]])


def test_a_feature_without_spread_is_centred_and_nothing_more() -> None:
    standardisation = Standardisation.fitted_on(ROWS)

    assert standardisation.scale[1] == 1.0
    assert standardisation.of(np.array([[3.0, 7.0, 3.0]]))[0, 1] == 2.0


def test_a_missing_statistic_is_read_at_the_fitting_mean() -> None:
    standardisation = Standardisation.fitted_on(ROWS)

    assert standardisation.centre[2] == 3.0
    assert standardisation.of(np.array([[3.0, 5.0, np.nan]]))[0, 2] == 0.0
    assert standardisation.of(ROWS)[1].tolist() == [0.0, 0.0, -1.0]
