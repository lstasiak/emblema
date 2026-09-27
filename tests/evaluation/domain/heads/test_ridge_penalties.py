import pytest

from emblema.evaluation.domain.exceptions import InvalidRidgePenaltiesError
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties


def test_a_fit_with_no_penalty_to_choose_is_refused() -> None:
    with pytest.raises(InvalidRidgePenaltiesError, match="at least one penalty"):
        RidgePenalties(())


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_a_penalty_that_is_not_positive_and_finite_is_refused(penalty: float) -> None:
    with pytest.raises(InvalidRidgePenaltiesError, match="positive and finite"):
        RidgePenalties((0.1, penalty))


@pytest.mark.parametrize("penalties", [(1.0, 0.1), (0.1, 0.1, 1.0)])
def test_penalties_out_of_order_or_repeated_are_refused(penalties: tuple[float, ...]) -> None:
    with pytest.raises(InvalidRidgePenaltiesError, match="ascending and distinct"):
        RidgePenalties(penalties)


def test_the_penalties_render_as_one_field() -> None:
    assert str(RidgePenalties((0.001, 1.0, 1000.0))) == "0.001 1 1000"
