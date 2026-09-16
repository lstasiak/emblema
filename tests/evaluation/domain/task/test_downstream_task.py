import pytest

from emblema.evaluation.domain.exceptions import FrozenTestSplitClosedError
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from tests.evaluation.support import TEST_SIDE, task, units


def test_a_task_names_the_units_a_budget_may_be_drawn_from() -> None:
    assert task().tuning_units == units("a", "b")
    assert task().validation_units == units("c")


def test_a_tuning_run_is_refused_the_frozen_side() -> None:
    with pytest.raises(FrozenTestSplitClosedError, match="frozen for a tuning run"):
        task().open_test_split(RunPurpose.TUNING)


def test_the_final_run_is_handed_the_frozen_side() -> None:
    assert task().open_test_split(RunPurpose.FINAL) == TEST_SIDE
