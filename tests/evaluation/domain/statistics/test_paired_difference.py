from math import nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedDifferenceError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind


def difference(
    reduction: float = 4.0,
    relative: float = 0.12,
    low: float = 1.0,
    high: float = 7.0,
    p_value: float = 0.01,
) -> PairedDifference:
    return PairedDifference(
        reduction=reduction,
        relative_reduction=relative,
        interval=BootstrapInterval(low=low, high=high, level=0.95),
        p_value=p_value,
    )


def test_the_claim_needs_both_the_share_and_the_interval() -> None:
    assert difference().confirms(0.10, ThresholdKind.RELATIVE)
    assert not difference(relative=0.08).confirms(0.10, ThresholdKind.RELATIVE)
    assert not difference(low=-0.5).confirms(0.10, ThresholdKind.RELATIVE)


def test_a_control_without_error_leaves_no_share_to_reach() -> None:
    # The control made no error, so the candidate can at best tie it; no share of nothing is a
    # reduction, and the absolute rule still reads the reduction itself.
    tied = PairedDifference(
        reduction=0.0,
        relative_reduction=None,
        interval=BootstrapInterval(low=0.0, high=0.0, level=0.95),
        p_value=1.0,
    )

    assert not tied.confirms(0.10, ThresholdKind.RELATIVE)
    assert not tied.confirms(0.01, ThresholdKind.ABSOLUTE)


def test_an_absolute_claim_reads_the_reduction_in_the_errors_unit_not_its_share() -> None:
    # A gain of 0.03 in area closes 12 % of a shortfall of 0.25: the absolute rule reads 0.03.
    gain = difference(reduction=0.03, relative=0.12, low=0.01, high=0.05)

    assert gain.confirms(0.02, ThresholdKind.ABSOLUTE)
    assert not gain.confirms(0.04, ThresholdKind.ABSOLUTE)
    assert not difference(reduction=0.03, low=-0.01).confirms(0.02, ThresholdKind.ABSOLUTE)


def test_a_difference_is_distinguishable_when_its_interval_keeps_zero_out() -> None:
    assert difference().distinguishable
    assert difference(reduction=-4.0, relative=-0.12, low=-7.0, high=-1.0).distinguishable
    assert not difference(low=-1.0).distinguishable


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reduction", nan, "reduction must be finite"),
        ("relative", nan, "relative_reduction must be finite"),
        ("p_value", 1.5, "p_value must lie in"),
        ("p_value", -0.1, "p_value must lie in"),
    ],
)
def test_a_difference_that_says_nothing_is_refused(field: str, value: float, message: str) -> None:
    with pytest.raises(InvalidPairedDifferenceError, match=message):
        difference(**{field: value})
