import pytest

from emblema.evaluation.domain.exceptions import InvalidOutcomeError, SingleClassSampleError
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from tests.evaluation.support import labelled

STRATA = ClassStrata()


def pool(positives: int = 14, negatives: int = 86) -> list[LabelledWindow]:
    return [
        labelled(f"stay-{index}", 0, 48.0, 1.0 if index < positives else 0.0)
        for index in range(positives + negatives)
    ]


def positives_in(windows: list[LabelledWindow], taken: tuple[int, ...]) -> int:
    return sum(1 for index in taken if windows[index].target == 1.0)


@pytest.mark.parametrize("wanted", [7, 20, 50, 99])
def test_every_budget_holds_each_outcome_within_one_window_of_its_share(wanted: int) -> None:
    windows = pool()

    taken = STRATA.draw(windows, wanted, seed=3)

    assert len(taken) == len(set(taken)) == wanted
    assert abs(positives_in(windows, taken) - 0.14 * wanted) < 1.0


def test_every_prefix_of_a_draw_holds_each_outcome_within_one_window_of_its_share() -> None:
    windows = pool()
    taken = STRATA.draw(windows, 100, seed=3)

    for size in range(1, 101):
        assert abs(positives_in(windows, taken[:size]) - 0.14 * size) < 1.0


def test_a_smaller_budget_is_carried_by_the_larger_one_at_the_same_seed() -> None:
    windows = pool()

    assert STRATA.draw(windows, 50, seed=3)[:20] == STRATA.draw(windows, 20, seed=3)


def test_the_order_the_pool_arrives_in_does_not_move_the_draw() -> None:
    windows = pool()
    reversed_windows = list(reversed(windows))

    forward = {windows[index].window for index in STRATA.draw(windows, 20, seed=3)}
    backward = {
        reversed_windows[index].window for index in STRATA.draw(reversed_windows, 20, seed=3)
    }

    assert forward == backward


def test_another_seed_draws_other_windows() -> None:
    windows = pool()

    assert set(STRATA.draw(windows, 20, seed=3)) != set(STRATA.draw(windows, 20, seed=4))


def test_a_budget_too_small_for_the_rarer_outcome_is_refused() -> None:
    # With one positive in seven, the fourth window is the first the positives are owed.
    windows = pool()

    with pytest.raises(SingleClassSampleError, match="one outcome only"):
        STRATA.draw(windows, 3, seed=3)
    assert positives_in(windows, STRATA.draw(windows, 4, seed=3)) == 1


def test_a_pool_of_one_outcome_is_refused() -> None:
    with pytest.raises(SingleClassSampleError, match="0 positive"):
        STRATA.draw(pool(positives=0, negatives=10), 5, seed=3)


def test_a_window_labelled_with_anything_but_an_outcome_is_refused() -> None:
    windows = [*pool(), labelled("stay-x", 0, 48.0, 0.5)]

    with pytest.raises(InvalidOutcomeError, match="stay-x"):
        STRATA.draw(windows, 10, seed=3)
