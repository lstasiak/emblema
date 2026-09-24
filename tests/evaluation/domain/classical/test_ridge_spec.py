import pytest

from emblema.evaluation.domain.classical.ridge_spec import RidgeSpec
from emblema.evaluation.domain.exceptions import InvalidRidgeSpecError


def test_a_fit_with_no_penalty_to_choose_is_refused() -> None:
    with pytest.raises(InvalidRidgeSpecError, match="at least one penalty"):
        RidgeSpec(penalties=(), threads=1)


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_a_penalty_that_is_not_positive_and_finite_is_refused(penalty: float) -> None:
    with pytest.raises(InvalidRidgeSpecError, match="positive and finite"):
        RidgeSpec(penalties=(penalty,), threads=1)


@pytest.mark.parametrize("penalties", [(10.0, 1.0), (1.0, 1.0)])
def test_penalties_out_of_order_or_repeated_are_refused(penalties: tuple[float, ...]) -> None:
    with pytest.raises(InvalidRidgeSpecError, match="ascending and distinct"):
        RidgeSpec(penalties=penalties, threads=1)


def test_a_fit_on_no_thread_is_refused() -> None:
    with pytest.raises(InvalidRidgeSpecError, match="threads"):
        RidgeSpec(penalties=(1.0,), threads=0)


def test_the_penalties_are_recorded_as_one_field_with_the_threads_beside_them() -> None:
    assert RidgeSpec(penalties=(0.001, 1.0, 1000.0), threads=2).parameters() == {
        "ridge_penalties": "0.001 1 1000",
        "threads": 2,
    }
