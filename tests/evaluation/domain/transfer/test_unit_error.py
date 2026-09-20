from math import inf

import pytest

from emblema.evaluation.domain.exceptions import InvalidUnitErrorError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.transfer.unit_error import UnitError
from tests.evaluation.support import prediction

A = UnitKey("a")


def test_a_units_error_sums_the_squared_errors_of_its_windows() -> None:
    error = UnitError.of(A, [prediction("a", 0, 10.0, 13.0), prediction("a", 1, 20.0, 16.0)])

    assert (error.squared_error, error.windows) == (25.0, 2)
    assert error.rmse == pytest.approx((25.0 / 2) ** 0.5)


def test_the_errors_per_unit_come_in_unit_order_whatever_order_the_answers_came_in() -> None:
    errors = UnitError.per_unit(
        (
            prediction("b", 2, 30.0, 30.0),
            prediction("a", 0, 10.0, 13.0),
            prediction("a", 1, 20.0, 16.0),
        )
    )

    assert [(str(e.unit), e.squared_error, e.windows) for e in errors] == [
        ("a", 25.0, 2),
        ("b", 0.0, 1),
    ]
    assert UnitError.per_unit(()) == ()


def test_a_prediction_of_another_unit_is_not_counted_against_this_one() -> None:
    with pytest.raises(InvalidUnitErrorError, match="counted against"):
        UnitError.of(A, [prediction("b", 0, 10.0, 13.0)])


def test_an_error_over_no_window_is_refused() -> None:
    with pytest.raises(InvalidUnitErrorError, match="must cover a window"):
        UnitError.of(A, [])


@pytest.mark.parametrize("squared_error", [-1.0, inf])
def test_a_sum_that_is_not_a_finite_non_negative_number_is_refused(squared_error: float) -> None:
    with pytest.raises(InvalidUnitErrorError, match="finite and not negative"):
        UnitError(unit=A, squared_error=squared_error, windows=1)
