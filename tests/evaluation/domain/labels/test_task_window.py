import pytest

from emblema.evaluation.domain.exceptions import InvalidTaskWindowError
from tests.evaluation.support import window


def test_a_window_says_whose_it_is_where_it_sits_and_when_it_ends() -> None:
    placed = window("engine-1", 17, 120.0)

    assert (str(placed.unit), placed.position, placed.ends_at) == ("engine-1", 17, 120.0)


def test_a_window_at_a_negative_position_is_refused() -> None:
    with pytest.raises(InvalidTaskWindowError, match="position"):
        window("engine-1", -1, 120.0)


@pytest.mark.parametrize("ends_at", [float("inf"), float("nan")])
def test_a_window_that_does_not_end_at_a_finite_moment_is_refused(ends_at: float) -> None:
    with pytest.raises(InvalidTaskWindowError, match="finite"):
        window("engine-1", 0, ends_at)
