import pytest

from emblema.evaluation.domain.exceptions import (
    InvalidComparisonRulesError,
    InvalidFamilyCorrectionError,
)
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor

RULES = ComparisonRules(
    minimum_relative_reduction=0.10,
    floor_share=0.03,
    correction=HolmCorrection(alpha=0.05),
    secondary_family_size=11,
)
FLOOR = PracticalFloor(value=1.5)


def difference(
    reduction: float, low: float, high: float, *, control: float = 40.0
) -> PairedDifference:
    return PairedDifference(
        reduction=reduction,
        relative_reduction=reduction / control,
        interval=BootstrapInterval(low=low, high=high, level=0.95),
        p_value=0.01 if low > 0.0 or high < 0.0 else 0.5,
    )


def test_the_endpoint_is_confirmed_by_the_share_the_interval_and_the_floor_together() -> None:
    assert RULES.endpoint_verdict(difference(6.0, 2.0, 10.0), FLOOR) is ComparisonVerdict.CONFIRMED


@pytest.mark.parametrize(
    ("reduction", "low", "high", "floor", "verdict"),
    [
        (6.0, -1.0, 13.0, 1.5, ComparisonVerdict.INDISTINGUISHABLE),
        (-6.0, -10.0, -2.0, 1.5, ComparisonVerdict.WORSE),
        (1.0, 0.2, 1.8, 1.5, ComparisonVerdict.PRACTICALLY_NIL),
        (3.0, 1.0, 5.0, 1.5, ComparisonVerdict.BELOW_REGISTERED_REDUCTION),
        # A tenth off, the whole interval above zero, and yet under a floor the control's own
        # spread over its repeats raised above the share: nil, not confirmed.
        (6.0, 2.0, 10.0, 7.0, ComparisonVerdict.PRACTICALLY_NIL),
    ],
)
def test_the_endpoint_that_falls_short_is_named_by_what_it_falls_short_of(
    reduction: float, low: float, high: float, floor: float, verdict: ComparisonVerdict
) -> None:
    found = RULES.endpoint_verdict(difference(reduction, low, high), PracticalFloor(value=floor))

    assert found is verdict


def test_a_secondary_cell_is_judged_by_the_familys_word_and_not_by_its_own_interval() -> None:
    kept_out = difference(6.0, 2.0, 10.0)

    assert (
        RULES.secondary_verdict(kept_out, FLOOR, rejected=False)
        is ComparisonVerdict.INDISTINGUISHABLE
    )
    assert (
        RULES.secondary_verdict(kept_out, FLOOR, rejected=True) is ComparisonVerdict.DISTINGUISHABLE
    )
    assert (
        RULES.secondary_verdict(difference(1.0, 0.2, 1.8), FLOOR, rejected=True)
        is ComparisonVerdict.PRACTICALLY_NIL
    )
    assert (
        RULES.secondary_verdict(difference(-6.0, -10.0, -2.0), FLOOR, rejected=True)
        is ComparisonVerdict.WORSE
    )


def test_the_secondary_family_is_the_registered_one_whatever_ran() -> None:
    # Three cells ran; 0.01 would be rejected among three and is not among eleven.
    assert RULES.secondary_rejections([0.01, 0.2, 0.3]) == (False, False, False)
    assert RULES.secondary_rejections([0.004, 0.2, 0.3]) == (True, False, False)
    with pytest.raises(InvalidFamilyCorrectionError, match="a family of 11 cannot hold 12"):
        RULES.secondary_rejections([0.01] * 12)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minimum_relative_reduction", 0.0, "minimum_relative_reduction must lie in"),
        ("minimum_relative_reduction", 1.0, "minimum_relative_reduction must lie in"),
        ("floor_share", -0.1, "floor_share must be finite"),
        ("secondary_family_size", 0, "secondary family needs a member"),
    ],
)
def test_rules_that_cannot_judge_are_refused(field: str, value: float, message: str) -> None:
    stated = {
        "minimum_relative_reduction": 0.10,
        "floor_share": 0.03,
        "correction": HolmCorrection(alpha=0.05),
        "secondary_family_size": 11,
        field: value,
    }
    with pytest.raises(InvalidComparisonRulesError, match=message):
        ComparisonRules(**stated)  # type: ignore[arg-type]
