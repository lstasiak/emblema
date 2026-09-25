from math import inf, nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidErrorOverRepeatsError
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats


def test_the_pooled_error_is_kept_and_the_spread_is_the_deviation_over_the_repeats() -> None:
    error = ErrorOverRepeats.of(40.0, [36.0, 44.0, 40.0])

    assert (error.pooled, error.repeats) == (40.0, 3)
    assert error.spread == pytest.approx(4.0)


def test_a_single_repeat_has_no_spread() -> None:
    assert ErrorOverRepeats.of(40.0, [41.0]) == ErrorOverRepeats(pooled=40.0, spread=0.0, repeats=1)


def test_it_reads_as_the_error_and_its_spread() -> None:
    assert str(ErrorOverRepeats.of(40.0, [36.0, 44.0, 40.0])) == "40 (spread 4 over 3 repeats)"
    assert str(ErrorOverRepeats.of(40.0, [40.0])) == "40 over one repeat"


def test_an_error_without_a_repeat_is_refused() -> None:
    with pytest.raises(InvalidErrorOverRepeatsError, match="needs a repeat"):
        ErrorOverRepeats.of(40.0, [])
    with pytest.raises(InvalidErrorOverRepeatsError, match="needs a repeat"):
        ErrorOverRepeats(pooled=40.0, spread=0.0, repeats=0)


@pytest.mark.parametrize("value", [-1.0, inf, nan])
def test_a_figure_that_is_not_a_finite_non_negative_number_is_refused(value: float) -> None:
    with pytest.raises(InvalidErrorOverRepeatsError, match="finite and not negative"):
        ErrorOverRepeats.of(value, [40.0])
    with pytest.raises(InvalidErrorOverRepeatsError, match="finite and not negative"):
        ErrorOverRepeats.of(40.0, [40.0, value])


def test_a_single_repeat_stated_with_a_spread_is_refused() -> None:
    with pytest.raises(InvalidErrorOverRepeatsError, match="single repeat has no spread"):
        ErrorOverRepeats(pooled=40.0, spread=1.0, repeats=1)
