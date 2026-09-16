import pytest

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError, UnlabelledWindowError
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme

SCHEME = RemainingLifeScheme(125.0)


def test_a_window_late_in_a_life_is_labelled_with_the_cycles_that_remain() -> None:
    assert SCHEME.target(failed_at=200.0, ends_at=120.0) == 80.0


def test_a_window_early_in_a_life_is_held_at_the_ceiling() -> None:
    assert SCHEME.target(failed_at=400.0, ends_at=120.0) == 125.0


def test_the_window_that_ends_at_the_failure_is_labelled_zero() -> None:
    assert SCHEME.target(failed_at=200.0, ends_at=200.0) == 0.0


def test_a_window_reaching_past_the_failure_carries_no_label() -> None:
    with pytest.raises(UnlabelledWindowError, match="reaches past the failure"):
        SCHEME.target(failed_at=200.0, ends_at=200.5)


@pytest.mark.parametrize("ceiling", [0.0, -1.0, float("inf"), float("nan")])
def test_a_ceiling_that_is_not_a_positive_number_is_refused(ceiling: float) -> None:
    with pytest.raises(InvalidLabelSchemeError, match="ceiling"):
        RemainingLifeScheme(ceiling)
