import pytest

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from tests.evaluation.support import TEST_SIDE, units


def test_the_three_sides_stand_apart() -> None:
    split = TaskSplit(tuning=units("a", "b"), validation=units("c"), test=TEST_SIDE)

    assert split.tuning == units("a", "b")
    assert split.validation == units("c")
    assert split.test.units == units("held/1")


def test_a_split_without_units_to_tune_on_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="tuning side"):
        TaskSplit(tuning=units(), validation=units("c"), test=TEST_SIDE)


def test_a_split_without_units_to_validate_on_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="validation side"):
        TaskSplit(tuning=units("a"), validation=units(), test=TEST_SIDE)


def test_a_unit_that_both_tunes_and_validates_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="tuning and validation"):
        TaskSplit(tuning=units("a", "b"), validation=units("b"), test=TEST_SIDE)


def test_a_unit_that_both_tunes_and_is_held_for_the_final_run_is_refused() -> None:
    held = FrozenTestSplit(units=units("a"), source="turbofans/test")

    with pytest.raises(InvalidTaskSplitError, match="tuning and test"):
        TaskSplit(tuning=units("a", "b"), validation=units("c"), test=held)


def test_a_unit_that_both_validates_and_is_held_for_the_final_run_is_refused() -> None:
    held = FrozenTestSplit(units=units("c"), source="turbofans/test")

    with pytest.raises(InvalidTaskSplitError, match="validation and test"):
        TaskSplit(tuning=units("a"), validation=units("c"), test=held)
