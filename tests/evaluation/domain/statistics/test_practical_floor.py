import pytest

from emblema.evaluation.domain.exceptions import InvalidPracticalFloorError
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor


def test_the_larger_of_the_share_and_the_spread_over_repeats_stands() -> None:
    quiet = PracticalFloor.of(ErrorOverRepeats.of(40.0, [40.1, 39.9, 40.0]), share=0.03)
    noisy = PracticalFloor.of(ErrorOverRepeats.of(40.0, [36.0, 44.0, 40.0]), share=0.03)

    assert quiet.value == pytest.approx(1.2)
    assert noisy.value == pytest.approx(4.0)


def test_a_single_repeat_has_no_spread_so_the_share_alone_stands() -> None:
    assert PracticalFloor.of(ErrorOverRepeats.of(40.0, [40.0]), share=0.03).value == pytest.approx(
        1.2
    )


def test_a_reduction_under_the_floor_is_swallowed_whichever_way_it_points() -> None:
    floor = PracticalFloor(value=1.2)

    assert [floor.swallows(reduction) for reduction in (1.1, -1.1, 1.2, -1.5)] == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("share", [-0.01, float("inf")])
def test_a_floor_without_a_share_is_refused(share: float) -> None:
    with pytest.raises(InvalidPracticalFloorError, match="share must be finite"):
        PracticalFloor.of(ErrorOverRepeats.of(40.0, [40.0]), share=share)


def test_a_negative_floor_is_refused() -> None:
    with pytest.raises(InvalidPracticalFloorError, match="not negative"):
        PracticalFloor(value=-0.1)
