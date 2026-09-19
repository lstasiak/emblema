import pytest

from emblema.evaluation.domain.exceptions import InvalidHolmCorrectionError
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection


def test_the_smallest_p_values_meet_the_strictest_levels_and_the_first_failure_stops() -> None:
    # Four comparisons at 5 %: 0.01 <= 0.05/4, 0.015 <= 0.05/3, 0.03 > 0.05/2 stops, and the
    # last would have passed its own level of 0.05 but stands with the one before it.
    rejected = HolmCorrection(alpha=0.05).rejected([0.03, 0.01, 0.04, 0.015])

    assert rejected == (False, True, False, True)


def test_a_family_of_one_is_tested_at_the_level_itself() -> None:
    assert HolmCorrection(alpha=0.05).rejected([0.049]) == (True,)
    assert HolmCorrection(alpha=0.05).rejected([0.051]) == (False,)


def test_a_family_where_every_comparison_holds_rejects_all_of_them() -> None:
    assert HolmCorrection(alpha=0.05).rejected([0.001, 0.002, 0.003]) == (True, True, True)


def test_a_family_where_none_holds_rejects_none() -> None:
    assert HolmCorrection(alpha=0.05).rejected([0.5, 0.6, 0.7]) == (False, False, False)


def test_a_family_larger_than_what_ran_holds_the_measured_to_the_whole_familys_levels() -> None:
    # 0.01 is rejected among three but not among eleven, where the smallest is held to 0.05/11;
    # the members that have not run count as never rejected and never loosen the levels.
    assert HolmCorrection(alpha=0.05).rejected([0.01, 0.2, 0.3]) == (True, False, False)
    assert HolmCorrection(alpha=0.05).rejected([0.01, 0.2, 0.3], family_size=11) == (
        False,
        False,
        False,
    )
    assert HolmCorrection(alpha=0.05).rejected([0.004, 0.0049], family_size=11) == (True, True)
    assert HolmCorrection(alpha=0.05).rejected([0.004], family_size=1) == (True,)


def test_a_family_smaller_than_the_comparisons_given_is_refused() -> None:
    with pytest.raises(InvalidHolmCorrectionError, match="a family of 2 cannot hold 3"):
        HolmCorrection().rejected([0.1, 0.2, 0.3], family_size=2)


def test_an_empty_family_or_a_p_value_out_of_range_is_refused() -> None:
    with pytest.raises(InvalidHolmCorrectionError, match="at least one comparison"):
        HolmCorrection().rejected([])
    with pytest.raises(InvalidHolmCorrectionError, match="p-value must lie in"):
        HolmCorrection().rejected([0.5, 1.5])


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_a_level_outside_the_open_unit_interval_is_refused(alpha: float) -> None:
    with pytest.raises(InvalidHolmCorrectionError, match="alpha must lie in"):
        HolmCorrection(alpha=alpha)
