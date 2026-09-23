"""Contract of the DownstreamTaskRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. What both owe is the split back exactly as it was
stored, the frozen side included: a split that came back different would move every number
measured before it moved.
"""

import pytest
from sqlalchemy import Engine

from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.domain.exceptions import TaskNotFoundError
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from tests.evaluation.support import FORECAST, TASK, task
from tests.support.database import clear_evaluation, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def tasks(request: pytest.FixtureRequest) -> DownstreamTaskRepository:
    if request.param == "in_memory":
        return InMemoryDownstreamTaskRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_evaluation(engine)
    return SqlAlchemyDownstreamTaskRepository(engine)


def test_a_task_comes_back_with_its_three_sides_as_they_were_stored(
    tasks: DownstreamTaskRepository,
) -> None:
    stated = task()

    tasks.save(stated)

    stored = tasks.get(TASK)
    assert stored.split == stated.split
    assert stored.manifest == stated.manifest


def test_a_forecasting_task_comes_back_reading_its_own_scheme(
    tasks: DownstreamTaskRepository,
) -> None:
    tasks.save(task(labels=FORECAST))

    assert tasks.get(TASK).label_scheme() == FORECAST


def test_a_task_whose_ceiling_was_stored_comes_back_under_it(
    tasks: DownstreamTaskRepository,
) -> None:
    stated = task()

    tasks.save(stated)

    assert tasks.get(TASK).label_scheme() == stated.label_scheme()
    assert tasks.get(TASK).stratification() == stated.stratification()


def test_saving_again_replaces_the_sides_rather_than_adding_to_them(
    tasks: DownstreamTaskRepository,
) -> None:
    tasks.save(task())

    tasks.save(task(labels=FORECAST))

    assert tasks.get(TASK).label_scheme() == FORECAST


def test_a_task_nobody_stored_is_refused(tasks: DownstreamTaskRepository) -> None:
    with pytest.raises(TaskNotFoundError):
        tasks.get(TASK)
