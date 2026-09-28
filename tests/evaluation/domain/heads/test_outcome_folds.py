import pytest

from emblema.evaluation.domain.exceptions import InvalidOutcomeFoldsError, UnfoldableOutcomesError
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds

# Fifty windows, seven of them positive, as a budget of fifty draws them from the stays.
OUTCOMES = [1.0 if place % 7 == 3 else 0.0 for place in range(50)]


def test_every_window_is_held_out_by_one_fold_and_fitted_on_by_every_other() -> None:
    folds = OutcomeFolds.of(OUTCOMES)

    held = [place for fold in range(folds.count) for place in folds.held_out(fold)]
    assert sorted(held) == list(range(len(OUTCOMES)))
    for fold in range(folds.count):
        assert set(folds.kept(fold)) == set(range(len(OUTCOMES))) - set(folds.held_out(fold))


def test_every_fold_holds_each_outcome_within_one_window_of_its_share() -> None:
    folds = OutcomeFolds.of(OUTCOMES)

    positives = [sum(OUTCOMES[place] for place in folds.held_out(f)) for f in range(folds.count)]
    sizes = [len(folds.held_out(fold)) for fold in range(folds.count)]
    assert folds.count == OutcomeFolds.COUNT
    assert max(positives) - min(positives) <= 1
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    "outcomes",
    [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
)
def test_every_head_fitted_without_a_fold_sees_both_outcomes(outcomes: list[float]) -> None:
    folds = OutcomeFolds.of(outcomes)

    for fold in range(folds.count):
        assert {outcomes[place] for place in folds.kept(fold)} == {0.0, 1.0}


def test_a_sample_smaller_than_the_folds_holds_out_one_window_at_a_time() -> None:
    folds = OutcomeFolds.of([0.0, 0.0, 1.0, 1.0])

    assert folds.assignment == (0, 1, 2, 3)


def test_the_folds_follow_the_samples_order_and_nothing_else() -> None:
    assert OutcomeFolds.of(OUTCOMES) == OutcomeFolds.of(list(OUTCOMES))


@pytest.mark.parametrize(
    ("outcomes", "message"),
    [
        ([0.0, 0.0, 0.0, 1.0], "at least 2 windows of each outcome, got 1 of outcome 1"),
        ([0.0, 1.0, 1.0, 1.0], "got 1 of outcome 0"),
        ([0.0, 0.0, 1.0, 0.5], "zero or one"),
    ],
)
def test_outcomes_no_folds_can_be_cut_from_are_refused(outcomes: list[float], message: str) -> None:
    with pytest.raises(UnfoldableOutcomesError, match=message):
        OutcomeFolds.of(outcomes)


@pytest.mark.parametrize("assignment", [(), (0, 2, 0), (1, 1, 2)])
def test_folds_not_numbered_from_zero_each_holding_a_window_are_refused(
    assignment: tuple[int, ...],
) -> None:
    with pytest.raises(InvalidOutcomeFoldsError, match="numbered from zero"):
        OutcomeFolds(assignment=assignment)
