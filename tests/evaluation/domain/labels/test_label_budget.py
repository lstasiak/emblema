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


def test_a_budget_reads_back_from_its_canonical_text() -> None:
    assert LabelBudget.parse("200") == LabelBudget.of(200)
    assert LabelBudget.parse("all") == LabelBudget.everything()


def test_a_budget_writes_itself_as_one_word() -> None:
    assert LabelBudget.of(200).text() == "200"
    assert LabelBudget.everything().text() == "all"


@pytest.mark.parametrize("text", ["", "every", "2.5", "-1"])
def test_text_that_names_no_budget_is_refused(text: str) -> None:
    with pytest.raises(InvalidLabelBudgetError):
        LabelBudget.parse(text)
