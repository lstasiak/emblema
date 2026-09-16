import pytest

from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.domain.exceptions import TaskNotFoundError
from tests.evaluation.support import TASK, task, units


def test_a_saved_task_comes_back_as_it_was_saved() -> None:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())

    assert tasks.get(TASK) == task()


def test_saving_again_replaces_the_earlier_state() -> None:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())
    tasks.save(task(tuning=units("a")))

    assert tasks.get(TASK).tuning_units == units("a")


def test_a_task_nobody_saved_is_not_found() -> None:
    with pytest.raises(TaskNotFoundError, match="no task stored"):
        InMemoryDownstreamTaskRepository().get(TASK)
