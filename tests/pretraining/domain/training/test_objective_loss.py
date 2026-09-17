from math import inf, nan

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.pretraining.domain.exceptions import InvalidObjectiveLossError
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
BOUNDED = ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.0)

errors = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def test_the_squared_reading_is_the_square_of_the_error() -> None:
    assert SQUARED.of_error(0.0) == 0.0
    assert SQUARED.of_error(-3.0) == 9.0
    assert SQUARED.of_error(100.0) == 10_000.0
    assert not SQUARED.is_bounded


def test_the_bounded_reading_is_a_square_within_the_knee_and_a_line_past_it() -> None:
    assert BOUNDED.of_error(0.0) == 0.0
    assert BOUNDED.of_error(0.5) == 0.125
    assert BOUNDED.of_error(-1.0) == 0.5
    assert BOUNDED.of_error(2.0) == 1.5
    assert BOUNDED.of_error(-100.0) == 99.5
    assert BOUNDED.is_bounded


def test_the_bounded_reading_meets_its_line_where_the_knee_is() -> None:
    wide = ObjectiveLoss(kind=LossKind.HUBER, huber_delta=3.0)

    below, at, above = wide.of_error(2.999), wide.of_error(3.0), wide.of_error(3.001)

    assert at == pytest.approx(0.5 * 9.0)
    assert below < at < above
    # The slope carries over: past the knee one more unit of error costs the knee itself.
    assert wide.of_error(13.0) - wide.of_error(12.0) == pytest.approx(3.0)


@given(error=errors)
def test_no_reading_is_ever_negative_and_a_bounded_one_never_exceeds_the_square(
    error: float,
) -> None:
    assert SQUARED.of_error(error) >= 0.0
    assert BOUNDED.of_error(error) >= 0.0
    assert BOUNDED.of_error(error) <= SQUARED.of_error(error)


@given(error=errors)
def test_a_reading_treats_a_miss_the_same_in_either_direction(error: float) -> None:
    assert SQUARED.of_error(error) == SQUARED.of_error(-error)
    assert BOUNDED.of_error(error) == BOUNDED.of_error(-error)


@pytest.mark.parametrize("delta", [0.0, -1.0, nan, inf])
def test_a_bounded_reading_without_a_knee_a_run_could_use_is_refused(delta: float) -> None:
    with pytest.raises(InvalidObjectiveLossError, match="huber_delta"):
        ObjectiveLoss(kind=LossKind.HUBER, huber_delta=delta)


def test_a_squared_reading_is_refused_a_knee_nobody_would_read() -> None:
    with pytest.raises(InvalidObjectiveLossError, match="no knee"):
        ObjectiveLoss(kind=LossKind.MSE, huber_delta=1.0)
