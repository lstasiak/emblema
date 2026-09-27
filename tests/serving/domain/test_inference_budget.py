import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.serving.domain.exceptions import InvalidInferenceBudgetError, WindowBeyondBudgetError
from emblema.serving.domain.inference_budget import InferenceBudget

BUDGET = InferenceBudget(windows=2, longest=100)


def test_the_budget_is_counted_in_windows_of_the_longest_length() -> None:
    assert BUDGET.ceiling == 10_000
    assert BUDGET.capacity == 20_000


@pytest.mark.parametrize(("windows", "longest"), [(0, 100), (2, 0), (-1, 100)])
def test_a_budget_counts_positive_windows_and_tokens(windows: int, longest: int) -> None:
    with pytest.raises(InvalidInferenceBudgetError):
        InferenceBudget(windows=windows, longest=longest)


def test_a_batch_costs_its_windows_padded_to_the_longest() -> None:
    assert BUDGET.cost(3, 40) == 3 * 40 * 40


def test_the_longest_windows_share_batches_and_none_is_padded_to_a_longer_one() -> None:
    lengths = [10, 100, 30, 100, 10]

    batches = BUDGET.plan(lengths, max_windows=16)

    assert batches[0] == (1,)
    assert batches[1] == (3,)
    assert sorted(position for batch in batches for position in batch) == [0, 1, 2, 3, 4]
    for batch in batches:
        assert lengths[batch[0]] == max(lengths[position] for position in batch)


def test_short_windows_fill_a_batch_up_to_what_a_long_one_would_cost() -> None:
    batches = BUDGET.plan([10] * 250, max_windows=1000)

    assert [len(batch) for batch in batches] == [100, 100, 50]


def test_a_batch_holds_at_most_the_windows_it_is_told() -> None:
    batches = BUDGET.plan([10] * 7, max_windows=3)

    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_a_window_that_alone_costs_more_than_a_batch_may_is_refused() -> None:
    with pytest.raises(WindowBeyondBudgetError, match="at most 100 tokens"):
        BUDGET.plan([10, 101], max_windows=4)


def test_no_windows_make_no_batches() -> None:
    assert BUDGET.plan([], max_windows=4) == ()


@given(
    lengths=st.lists(st.integers(min_value=1, max_value=100), max_size=60),
    max_windows=st.integers(min_value=1, max_value=20),
)
def test_every_window_is_in_exactly_one_batch_and_no_batch_costs_too_much(
    lengths: list[int], max_windows: int
) -> None:
    batches = BUDGET.plan(lengths, max_windows=max_windows)

    assert sorted(position for batch in batches for position in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= max_windows
        assert BUDGET.cost(len(batch), lengths[batch[0]]) <= BUDGET.ceiling
        assert lengths[batch[0]] == max(lengths[position] for position in batch)
