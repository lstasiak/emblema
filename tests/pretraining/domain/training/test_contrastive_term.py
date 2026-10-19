from math import inf, log, nan

import pytest

from emblema.pretraining.domain.exceptions import InvalidContrastiveTermError
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm

NONE = ContrastiveTerm.none()
STATED = ContrastiveTerm.of(weight=0.5, temperature=0.2)


def test_no_term_weighs_nothing_and_reads_no_temperature() -> None:
    assert ContrastiveTerm() == NONE
    assert not NONE.is_stated
    assert (NONE.weight, NONE.temperature) == (0.0, 0.0)


def test_a_stated_term_carries_its_weight_and_temperature() -> None:
    assert STATED.is_stated
    assert (STATED.weight, STATED.temperature) == (0.5, 0.2)


def test_a_term_of_no_weight_is_refused_a_temperature_nobody_would_read() -> None:
    with pytest.raises(InvalidContrastiveTermError, match="no weight"):
        ContrastiveTerm(weight=0.0, temperature=0.2)


@pytest.mark.parametrize("weight", [-0.5, nan, inf])
def test_a_weight_that_is_negative_or_not_a_number_is_refused(weight: float) -> None:
    with pytest.raises(InvalidContrastiveTermError, match="weight"):
        ContrastiveTerm.of(weight=weight, temperature=0.2)


@pytest.mark.parametrize("temperature", [0.0, -0.1, nan, inf])
def test_a_stated_term_needs_a_positive_finite_temperature(temperature: float) -> None:
    with pytest.raises(InvalidContrastiveTermError, match="temperature"):
        ContrastiveTerm.of(weight=0.5, temperature=temperature)


def test_the_chance_cost_of_an_anchor_is_the_log_of_the_candidates_it_has() -> None:
    assert ContrastiveTerm.chance_of(16) == pytest.approx(log(31))
    assert ContrastiveTerm.chance_of(2) == pytest.approx(log(3))


def test_one_window_or_none_has_no_distractor_and_costs_nothing() -> None:
    assert ContrastiveTerm.chance_of(1) == 0.0
    assert ContrastiveTerm.chance_of(0) == 0.0


def test_a_negative_count_of_windows_is_refused() -> None:
    with pytest.raises(InvalidContrastiveTermError, match="-1 windows"):
        ContrastiveTerm.chance_of(-1)
