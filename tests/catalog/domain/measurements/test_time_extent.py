import pytest

from emblema.catalog.domain.exceptions import InvalidTimeExtentError
from emblema.catalog.domain.measurements.time_extent import TimeExtent


@pytest.mark.parametrize(("start", "end"), [(1.0, 1.0), (2.0, 1.0)])
def test_extent_start_precedes_end(start: float, end: float) -> None:
    with pytest.raises(InvalidTimeExtentError, match="precede"):
        TimeExtent(start, end)


@pytest.mark.parametrize(("start", "end"), [(float("nan"), 1.0), (0.0, float("inf"))])
def test_extent_bounds_are_finite(start: float, end: float) -> None:
    with pytest.raises(InvalidTimeExtentError, match="finite"):
        TimeExtent(start, end)


def test_extent_is_half_open() -> None:
    extent = TimeExtent(1.0, 4.0)

    assert extent.length == 3.0
    assert extent.contains(1.0)
    assert extent.contains(3.999)
    assert not extent.contains(4.0)
    assert not extent.contains(0.999)
