import math

import pytest

from emblema.catalog.domain.cleaning.value_bounds import ValueBounds
from emblema.catalog.domain.exceptions import InvalidValueBoundsError

PH = ValueBounds(
    channel="pH",
    removed_below=6.3,
    clipped_below=6.5,
    clipped_above=8.4,
    removed_above=10.0,
)


@pytest.mark.parametrize("value", [735.0, 94.0, 1.0, 3.0, 6.29, 10.01])
def test_a_value_no_unit_can_have_is_removed(value: float) -> None:
    assert PH.cleaned(value) is None


@pytest.mark.parametrize(("value", "kept"), [(6.4, 6.5), (9.0, 8.4), (6.3, 6.5), (10.0, 8.4)])
def test_an_extreme_but_possible_value_is_clipped_to_the_inner_bound(
    value: float, kept: float
) -> None:
    assert PH.cleaned(value) == kept


@pytest.mark.parametrize("value", [6.5, 7.38, 8.4])
def test_a_value_inside_the_inner_bounds_is_kept_as_it_is(value: float) -> None:
    assert PH.cleaned(value) == value


@pytest.mark.parametrize(
    "bounds",
    [(6.5, 6.3, 8.4, 10.0), (6.3, 6.5, 10.0, 8.4), (6.3, 8.5, 8.4, 10.0)],
    ids=["removed above clipped below", "clipped above removed above", "inner pair crossed"],
)
def test_bounds_out_of_order_are_refused(bounds: tuple[float, float, float, float]) -> None:
    with pytest.raises(InvalidValueBoundsError, match="low to high"):
        ValueBounds(
            channel="pH",
            removed_below=bounds[0],
            clipped_below=bounds[1],
            clipped_above=bounds[2],
            removed_above=bounds[3],
        )


@pytest.mark.parametrize("bound", [math.nan, math.inf])
def test_a_bound_that_is_not_finite_is_refused(bound: float) -> None:
    with pytest.raises(InvalidValueBoundsError, match="finite"):
        ValueBounds(
            channel="pH",
            removed_below=0.0,
            clipped_below=0.0,
            clipped_above=1.0,
            removed_above=bound,
        )


@pytest.mark.parametrize("channel", ["", " pH", "pH "])
def test_a_blank_or_padded_channel_name_is_refused(channel: str) -> None:
    with pytest.raises(InvalidValueBoundsError, match="channel name"):
        ValueBounds(
            channel=channel,
            removed_below=0.0,
            clipped_below=0.0,
            clipped_above=1.0,
            removed_above=1.0,
        )
