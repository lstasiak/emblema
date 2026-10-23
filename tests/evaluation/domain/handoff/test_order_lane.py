import pytest

from emblema.evaluation.domain.exceptions import InvalidOrderLaneError
from emblema.evaluation.domain.handoff.order_lane import OrderLane


def test_a_lane_takes_every_count_th_position_from_its_index() -> None:
    second_of_three = OrderLane(index=1, count=3)

    assert [p for p in range(8) if second_of_three.takes(p)] == [1, 4, 7]
    assert all(OrderLane(index=0, count=1).takes(p) for p in range(8))


def test_the_lanes_of_an_order_take_every_position_once() -> None:
    lanes = [OrderLane(index=i, count=4) for i in range(4)]

    for position in range(10):
        assert sum(lane.takes(position) for lane in lanes) == 1


def test_a_lane_is_read_as_it_is_written() -> None:
    assert OrderLane.parse("2/5") == OrderLane(index=2, count=5)
    assert OrderLane.parse(" 0/1 ") == OrderLane(index=0, count=1)
    assert str(OrderLane(index=2, count=5)) == "2/5"


@pytest.mark.parametrize(
    "text", ["2", "2/", "/5", "two/5", "2/5/1", "1.5/4", "", "+1/2", "1_0/20", "1 / 2", "-1/2"]
)
def test_a_lane_not_written_index_slash_count_is_refused(text: str) -> None:
    with pytest.raises(InvalidOrderLaneError, match="index/count"):
        OrderLane.parse(text)


@pytest.mark.parametrize(("index", "count"), [(0, 0), (0, -1), (-1, 3), (3, 3), (4, 3)])
def test_a_lane_outside_its_count_is_refused(index: int, count: int) -> None:
    with pytest.raises(InvalidOrderLaneError):
        OrderLane(index=index, count=count)
    with pytest.raises(InvalidOrderLaneError):
        OrderLane.parse(f"{index}/{count}")
