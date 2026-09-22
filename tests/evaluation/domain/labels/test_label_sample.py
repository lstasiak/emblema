import pytest

from emblema.evaluation.domain.exceptions import EmptyLabelSampleError, InvalidLabelBudgetError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_bins import TargetBins
from tests.evaluation.support import TASK, window

BINS = TargetBins(3)


def pool(size: int = 12) -> list[LabelledWindow]:
    return [
        LabelledWindow(
            window=window(f"engine-{position % 4}", position, float(position)),
            target=float(position),
        )
        for position in range(size)
    ]


def drawn(budget: LabelBudget, seed: int, windows: list[LabelledWindow] | None = None) -> list[int]:
    sample = LabelSample.drawn(TASK, windows or pool(), budget, BINS, seed)
    return [labelled.window.position for labelled in sample.windows]


def test_the_same_budget_and_seed_draw_the_same_windows() -> None:
    assert drawn(LabelBudget.of(6), seed=3) == drawn(LabelBudget.of(6), seed=3)


def test_another_seed_draws_other_windows() -> None:
    assert drawn(LabelBudget.of(6), seed=3) != drawn(LabelBudget.of(6), seed=4)


def test_the_order_the_pool_arrives_in_does_not_move_the_draw() -> None:
    assert drawn(LabelBudget.of(6), seed=3, windows=list(reversed(pool()))) == drawn(
        LabelBudget.of(6), seed=3
    )


def test_a_budget_is_spread_evenly_over_the_strata_of_the_target() -> None:
    positions = drawn(LabelBudget.of(6), seed=3)

    # Twelve windows with targets 0..11 make strata 0-3, 4-7 and 8-11.
    assert [
        sum(1 for position in positions if position // 4 == stratum) for stratum in range(3)
    ] == [
        2,
        2,
        2,
    ]


def test_a_budget_that_does_not_divide_by_the_strata_differs_by_one_at_most() -> None:
    positions = drawn(LabelBudget.of(7), seed=3)
    per_stratum = [
        sum(1 for position in positions if position // 4 == stratum) for stratum in range(3)
    ]

    assert max(per_stratum) - min(per_stratum) == 1
    assert sum(per_stratum) == 7


def test_the_largest_budget_is_every_window_the_pool_holds() -> None:
    assert sorted(drawn(LabelBudget.everything(), seed=3)) == list(range(12))


def test_the_largest_budget_needs_no_strata_to_fill() -> None:
    # Nothing is spread when everything is taken, so a pool smaller than the stratification is
    # still allowed the one budget it can serve.
    tiny = pool(2)

    sample = LabelSample.drawn(TASK, tiny, LabelBudget.everything(), BINS, seed=3)

    assert len(sample.windows) == 2


def test_a_smaller_budget_is_carried_by_the_larger_one_at_the_same_seed() -> None:
    # A curve compares budgets, so a larger budget that dropped windows the smaller one had would
    # be comparing two samples as well as two budgets.
    assert set(drawn(LabelBudget.of(3), seed=3)) <= set(drawn(LabelBudget.of(6), seed=3))


def test_a_budget_larger_than_the_pool_is_refused() -> None:
    with pytest.raises(InvalidLabelBudgetError, match="exceeds"):
        drawn(LabelBudget.of(13), seed=3)


def test_the_sample_carries_what_drew_it() -> None:
    sample = LabelSample.drawn(TASK, pool(), LabelBudget.of(6), BINS, seed=3)

    assert sample.task == TASK
    assert sample.seed == 3
    assert sample.budget == LabelBudget.of(6)


def test_the_sample_counts_the_units_its_windows_came_from() -> None:
    # Twelve windows over four engines: the whole pool reaches every engine, a budget of six
    # drawn across the strata happens to reach three of them under this seed, one window one.
    assert LabelSample.drawn(TASK, pool(), LabelBudget.everything(), BINS, seed=3).unit_count == 4
    assert LabelSample.drawn(TASK, pool(), LabelBudget.of(6), BINS, seed=3).unit_count == 3
    assert LabelSample.drawn(TASK, pool(), LabelBudget.of(1), BINS, seed=3).unit_count == 1


def test_the_sample_knows_the_mean_of_its_labels() -> None:
    sample = LabelSample.drawn(TASK, pool(), LabelBudget.of(6), BINS, seed=3)

    expected = sum(labelled.target for labelled in sample.windows) / 6
    assert sample.mean_target == pytest.approx(expected)
    assert (
        LabelSample.drawn(TASK, pool(), LabelBudget.everything(), BINS, seed=3).mean_target == 5.5
    )


def test_a_sample_without_a_window_has_no_mean_label() -> None:
    empty = LabelSample(task=TASK, windows=(), budget=LabelBudget.of(1), seed=1)

    with pytest.raises(EmptyLabelSampleError):
        _ = empty.mean_target
