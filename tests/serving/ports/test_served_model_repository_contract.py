"""Contract of the ServedModelRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. Both give a model back as it was stored, withdrawn
or not, and both refuse a second model serving an artifact already in service — the database
adapter by a partial unique index, which is where the rule has to hold for two processes at once.
"""

from uuid import UUID

import pytest
from sqlalchemy import Engine

from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.serving.domain.exceptions import ArtifactAlreadyServedError, ServedModelNotFoundError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.served_model_repository import ServedModelRepository
from tests.serving.support import MODEL, WITHDRAWN, served
from tests.support.database import clear_serving, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]
SECOND = ServedModelId(UUID(int=99))


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def models(request: pytest.FixtureRequest) -> ServedModelRepository:
    if request.param == "in_memory":
        return InMemoryServedModelRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_serving(engine)
    return SqlAlchemyServedModelRepository(engine)


def test_a_model_in_service_comes_back_as_it_was_stored(models: ServedModelRepository) -> None:
    models.save(served())

    assert models.get(MODEL) == served()


def test_a_withdrawn_model_comes_back_withdrawn(models: ServedModelRepository) -> None:
    models.save(served())
    withdrawn = served().withdraw(WITHDRAWN)

    models.save(withdrawn)

    assert models.get(MODEL) == withdrawn


def test_a_model_nobody_stored_is_not_found(models: ServedModelRepository) -> None:
    with pytest.raises(ServedModelNotFoundError):
        models.get(MODEL)


def test_a_second_model_serving_an_artifact_already_in_service_is_refused(
    models: ServedModelRepository,
) -> None:
    models.save(served())

    with pytest.raises(ArtifactAlreadyServedError):
        models.save(served(served_model_id=SECOND))

    with pytest.raises(ServedModelNotFoundError):
        models.get(SECOND)


def test_an_artifact_withdrawn_from_service_may_be_served_by_another_model(
    models: ServedModelRepository,
) -> None:
    models.save(served().withdraw(WITHDRAWN))

    models.save(served(served_model_id=SECOND))

    assert models.get(SECOND).artifact == models.get(MODEL).artifact
