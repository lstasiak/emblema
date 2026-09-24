from math import inf, nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidWindowPredictionError
from tests.evaluation.support import prediction


def test_the_squared_error_is_the_squared_distance_to_the_target() -> None:
    assert prediction("a", 0, target=10.0, predicted=13.0).squared_error == 9.0


@pytest.mark.parametrize(("target", "predicted"), [(nan, 1.0), (1.0, inf), (-inf, 1.0)])
def test_a_prediction_that_is_not_a_number_is_refused(target: float, predicted: float) -> None:
    with pytest.raises(InvalidWindowPredictionError, match="must be finite"):
        prediction("a", 0, target=target, predicted=predicted)
