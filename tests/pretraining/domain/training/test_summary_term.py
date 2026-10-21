from math import inf, nan

import pytest

from emblema.pretraining.domain.exceptions import InvalidSummaryTermError
from emblema.pretraining.domain.training.summary_term import SUMMARIES, SummaryTerm


def test_no_term_weighs_nothing() -> None:
    assert SummaryTerm() == SummaryTerm.none()
    assert not SummaryTerm.none().is_stated


def test_a_stated_term_carries_its_weight() -> None:
    term = SummaryTerm.of(weight=0.5)
    assert term.is_stated
    assert term.weight == 0.5


@pytest.mark.parametrize("weight", [-0.5, nan, inf])
def test_a_weight_that_is_negative_or_not_a_number_is_refused(weight: float) -> None:
    with pytest.raises(InvalidSummaryTermError, match="weight"):
        SummaryTerm.of(weight=weight)


def test_the_summaries_are_the_level_and_the_extremes() -> None:
    assert SUMMARIES == ("mean", "minimum", "maximum")
