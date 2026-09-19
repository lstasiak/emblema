from math import inf

import pytest

from emblema.evaluation.domain.exceptions import InvalidPracticalFloorError
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor


def test_the_larger_of_the_share_and_the_spread_over_repeats_stands() -> None:
    quiet = PracticalFloor.of(40.0, [40.1, 39.9, 40.0], share=0.03)
    noisy = PracticalFloor.of(40.0, [36.0, 44.0, 40.0], share=0.03)

    assert quiet.value == pytest.approx(1.2)
    assert noisy.value == pytest.approx(4.0)


def test_a_single_repeat_has_no_spread_so_the_share_alone_stands() -> None:
    assert PracticalFloor.of(40.0, [40.0], share=0.03).value == pytest.approx(1.2)


def test_a_reduction_under_the_floor_is_swallowed_whichever_way_it_points() -> None:
    floor = PracticalFloor(value=1.2)

    assert [floor.swallows(reduction) for reduction in (1.1, -1.1, 1.2, -1.5)] == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("rmse", "repeats", "share", "message"),
    [
        (40.0, [40.0], -0.01, "share must be finite"),
        (-1.0, [40.0], 0.03, "control_rmse must be finite"),
        (inf, [40.0], 0.03, "control_rmse must be finite"),
        (40.0, [], 0.03, "needs the control's repeats"),
    ],
)
def test_a_floor_without_a_control_or_a_share_is_refused(
    rmse: float, repeats: list[float], share: float, message: str
) -> None:
    with pytest.raises(InvalidPracticalFloorError, match=message):
        PracticalFloor.of(rmse, repeats, share=share)


def test_a_negative_floor_is_refused() -> None:
    with pytest.raises(InvalidPracticalFloorError, match="not negative"):
        PracticalFloor(value=-0.1)
