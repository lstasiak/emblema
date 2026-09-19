from math import inf

import pytest

from emblema.evaluation.domain.exceptions import InvalidBootstrapIntervalError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval


def interval(low: float, high: float) -> BootstrapInterval:
    return BootstrapInterval(low=low, high=high, level=0.95)


def test_an_interval_says_which_side_of_zero_it_lies_on() -> None:
    above, below, across = interval(1.0, 3.0), interval(-3.0, -1.0), interval(-1.0, 1.0)

    assert (above.above_zero, above.excludes_zero) == (True, True)
    assert (below.above_zero, below.excludes_zero) == (False, True)
    assert (across.above_zero, across.excludes_zero) == (False, False)


def test_an_interval_inside_a_margin_is_an_equivalence() -> None:
    assert interval(-0.5, 0.4).within(0.5)
    assert not interval(-0.6, 0.4).within(0.5)
    assert not interval(0.1, 0.6).within(0.5)


def test_an_interval_renders_with_its_signs() -> None:
    assert str(interval(-1.25, 2.5)) == "[-1.250, +2.500]"


@pytest.mark.parametrize(
    ("low", "high", "level", "message"),
    [
        (2.0, 1.0, 0.95, "finite ends in order"),
        (-inf, 1.0, 0.95, "finite ends in order"),
        (0.0, 1.0, 1.0, "level must lie in"),
        (0.0, 1.0, 0.0, "level must lie in"),
    ],
)
def test_an_interval_out_of_order_or_at_no_level_is_refused(
    low: float, high: float, level: float, message: str
) -> None:
    with pytest.raises(InvalidBootstrapIntervalError, match=message):
        BootstrapInterval(low=low, high=high, level=level)
