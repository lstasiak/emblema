import pytest

from emblema.evaluation.domain.exceptions import InvalidLabelBudgetError
from emblema.evaluation.domain.labels.label_budget import LabelBudget


def test_a_counted_budget_takes_what_it_asks_for() -> None:
    assert LabelBudget.of(50).drawn_from(2651) == 50


def test_the_largest_budget_takes_whatever_the_pool_holds() -> None:
    assert LabelBudget.everything().drawn_from(2651) == 2651


def test_a_budget_larger_than_the_pool_is_refused_rather_than_quietly_shrunk() -> None:
    with pytest.raises(InvalidLabelBudgetError, match="exceeds"):
        LabelBudget.of(3000).drawn_from(2651)


@pytest.mark.parametrize("windows", [0, -1])
def test_a_budget_of_no_windows_is_refused(windows: int) -> None:
    with pytest.raises(InvalidLabelBudgetError, match="at least one window"):
        LabelBudget.of(windows)
