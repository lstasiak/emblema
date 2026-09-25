import pytest

from emblema.evaluation.domain.exceptions import InvalidFamilyCorrectionError
from emblema.evaluation.domain.statistics.benjamini_hochberg_correction import (
    BenjaminiHochbergCorrection,
)
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection

# The worked example of Benjamini and Hochberg (1995), table 1: fifteen p-values, of which the
# procedure rejects the first four at a share of 5 %, where Bonferroni would reject three.
TEXTBOOK = (
    0.0001,
    0.0004,
    0.0019,
    0.0095,
    0.0201,
    0.0278,
    0.0298,
    0.0344,
    0.0459,
    0.3240,
    0.4262,
    0.5719,
    0.6528,
    0.7590,
    1.0000,
)


def test_the_textbook_family_rejects_the_first_four() -> None:
    rejected = BenjaminiHochbergCorrection(alpha=0.05).rejected(TEXTBOOK)

    assert rejected == (True,) * 4 + (False,) * 11


def test_a_rank_that_holds_rejects_everything_ranked_before_it_even_where_a_step_failed() -> None:
    # 0.04 fails its own level of 0.05 * 3 / 4 but 0.045 at rank four holds 0.05, so all four go.
    rejected = BenjaminiHochbergCorrection(alpha=0.05).rejected([0.04, 0.01, 0.045, 0.02])

    assert rejected == (True, True, True, True)


def test_it_rejects_no_fewer_than_holm_on_the_same_family() -> None:
    families = ([0.03, 0.01, 0.04, 0.015], TEXTBOOK, [0.5, 0.6, 0.7], [0.001, 0.002, 0.003])

    for family in families:
        holm = HolmCorrection(alpha=0.05).rejected(family)
        step_up = BenjaminiHochbergCorrection(alpha=0.05).rejected(family)
        assert all(
            rejected_by_holm <= by_step_up
            for rejected_by_holm, by_step_up in zip(holm, step_up, strict=True)
        )


def test_a_family_of_one_is_tested_at_the_level_itself() -> None:
    assert BenjaminiHochbergCorrection(alpha=0.05).rejected([0.049]) == (True,)
    assert BenjaminiHochbergCorrection(alpha=0.05).rejected([0.051]) == (False,)


def test_a_family_larger_than_what_ran_holds_the_measured_to_ranks_over_the_whole_family() -> None:
    # Among three, 0.03 at rank three holds 0.05; among eleven, rank three is held to 0.05 * 3 / 11.
    assert BenjaminiHochbergCorrection(alpha=0.05).rejected([0.01, 0.02, 0.03]) == (
        True,
        True,
        True,
    )
    assert BenjaminiHochbergCorrection(alpha=0.05).rejected([0.01, 0.02, 0.03], family_size=11) == (
        False,
        False,
        False,
    )
    assert BenjaminiHochbergCorrection(alpha=0.05).rejected([0.004], family_size=11) == (True,)


def test_a_family_smaller_than_the_comparisons_given_is_refused() -> None:
    with pytest.raises(InvalidFamilyCorrectionError, match="a family of 2 cannot hold 3"):
        BenjaminiHochbergCorrection().rejected([0.1, 0.2, 0.3], family_size=2)


def test_an_empty_family_or_a_p_value_out_of_range_is_refused() -> None:
    with pytest.raises(InvalidFamilyCorrectionError, match="at least one comparison"):
        BenjaminiHochbergCorrection().rejected([])
    with pytest.raises(InvalidFamilyCorrectionError, match="p-value must lie in"):
        BenjaminiHochbergCorrection().rejected([0.5, 1.5])


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_a_level_outside_the_open_unit_interval_is_refused(alpha: float) -> None:
    with pytest.raises(InvalidFamilyCorrectionError, match="alpha must lie in"):
        BenjaminiHochbergCorrection(alpha=alpha)


def test_the_two_corrections_are_told_apart_by_name() -> None:
    assert (HolmCorrection.NAME, BenjaminiHochbergCorrection.NAME) == (
        "holm",
        "benjamini_hochberg",
    )
