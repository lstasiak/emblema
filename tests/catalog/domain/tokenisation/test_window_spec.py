import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.exceptions import InvalidWindowSpecError
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec


def windows(spec: WindowSpec, extent: TimeExtent) -> list[TimeExtent]:
    return list(spec.windows_over(extent))


@pytest.mark.parametrize(("length", "stride"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_length_and_stride_are_positive(length: float, stride: float) -> None:
    with pytest.raises(InvalidWindowSpecError, match="positive"):
        WindowSpec(length, stride)


@pytest.mark.parametrize(
    ("length", "stride"), [(float("inf"), 1.0), (1.0, float("nan"))], ids=["length", "stride"]
)
def test_length_and_stride_are_finite(length: float, stride: float) -> None:
    with pytest.raises(InvalidWindowSpecError, match="finite"):
        WindowSpec(length, stride)


def test_an_engine_of_128_cycles_holds_16_windows_of_50_every_5() -> None:
    laid = windows(WindowSpec(50, 5), TimeExtent(1.0, 129.0))

    assert len(laid) == 16
    assert laid[0] == TimeExtent(1.0, 51.0)
    assert laid[-1] == TimeExtent(76.0, 126.0)


def test_windows_start_at_the_extent_start_and_advance_by_the_stride() -> None:
    assert windows(WindowSpec(2, 1.5), TimeExtent(10.0, 15.0)) == [
        TimeExtent(10.0, 12.0),
        TimeExtent(11.5, 13.5),
        TimeExtent(13.0, 15.0),
    ]


def test_a_window_that_fits_exactly_counts_and_the_next_does_not() -> None:
    assert windows(WindowSpec(4, 1), TimeExtent(0.0, 4.0)) == [TimeExtent(0.0, 4.0)]


def test_an_extent_shorter_than_a_window_holds_none() -> None:
    assert windows(WindowSpec(5, 1), TimeExtent(0.0, 4.9)) == []


@given(
    span=st.integers(min_value=1, max_value=500),
    length=st.integers(min_value=1, max_value=100),
    stride=st.integers(min_value=1, max_value=50),
)
def test_the_count_is_the_closed_form_of_the_data_spike(
    span: int, length: int, stride: int
) -> None:
    laid = windows(WindowSpec(length, stride), TimeExtent(1.0, 1.0 + span))

    expected = 0 if span < length else math.floor((span - length) / stride) + 1
    assert len(laid) == expected
    assert all(window.length == length for window in laid)
    assert all(window.start >= 1.0 and window.end <= 1.0 + span for window in laid)
