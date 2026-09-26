"""Contract of the ServedModelListing port, run against every adapter.

Both adapters order and cut the same way, so a page read from the database is the page read
from memory; the database adapter needs the metadata database and is marked ``integration``.
"""

from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import Engine

from emblema.serving.adapters.in_memory.served_model_listing import InMemoryServedModelListing
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.adapters.persistence.served_model_listing import SqlAlchemyServedModelListing
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_position import ServedModelPosition
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.serving.ports.served_model_listing import ServedModelListing
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.serving.support import PROMOTED, WITHDRAWN, served
from tests.support.database import clear_serving, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]


def model(number: int, *, hours: int, withdrawn: bool = False) -> ServedModel:
    stated = served(
        served_model_id=ServedModelId(UUID(int=number)),
        artifact=ArtifactRef(key=f"durable/{number}", checksum=Checksum.of_bytes(bytes([number]))),
        promoted_at=UtcDateTime(PROMOTED.value + timedelta(hours=hours)),
    )
    return (
        stated.withdraw(UtcDateTime(WITHDRAWN.value + timedelta(hours=hours)))
        if withdrawn
        else stated
    )


# Four models: two promoted at the same instant, to hold the tie-break, and one withdrawn.
MODELS = (
    model(1, hours=1),
    model(2, hours=2, withdrawn=True),
    model(3, hours=3),
    model(4, hours=3),
)
NEWEST_FIRST = (MODELS[2], MODELS[3], MODELS[1], MODELS[0])


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def listing(request: pytest.FixtureRequest) -> ServedModelListing:
    if request.param == "in_memory":
        repository = InMemoryServedModelRepository()
        for stored in MODELS:
            repository.save(stored)
        return InMemoryServedModelListing(repository)
    engine: Engine = request.getfixturevalue("database")
    clear_serving(engine)
    for stored in MODELS:
        SqlAlchemyServedModelRepository(engine).save(stored)
    return SqlAlchemyServedModelListing(engine)


def test_the_first_page_holds_the_newest_promotions_identity_breaking_a_tie(
    listing: ServedModelListing,
) -> None:
    assert listing.page(after=None, limit=10, state=None) == NEWEST_FIRST


def test_a_page_continues_after_the_model_named(listing: ServedModelListing) -> None:
    first = listing.page(after=None, limit=2, state=None)

    second = listing.page(after=ServedModelPosition.of(first[-1]), limit=2, state=None)

    assert first == NEWEST_FIRST[:2]
    assert second == NEWEST_FIRST[2:]


def test_a_page_is_cut_at_the_limit(listing: ServedModelListing) -> None:
    assert listing.page(after=None, limit=1, state=None) == NEWEST_FIRST[:1]


def test_a_page_can_be_narrowed_to_one_state(listing: ServedModelListing) -> None:
    serving = listing.page(after=None, limit=10, state=ServedModelState.SERVING)
    withdrawn = listing.page(after=None, limit=10, state=ServedModelState.WITHDRAWN)

    assert serving == (MODELS[2], MODELS[3], MODELS[0])
    assert withdrawn == (MODELS[1],)


def test_a_page_continues_after_a_position_no_stored_model_holds(
    listing: ServedModelListing,
) -> None:
    # Half an hour before the latest promotions, under an identity nobody holds.
    between = ServedModelPosition(
        promoted_at=UtcDateTime(MODELS[2].promoted_at.value - timedelta(minutes=30)),
        served_model_id=ServedModelId(UUID(int=99)),
    )

    assert listing.page(after=between, limit=10, state=None) == NEWEST_FIRST[2:]


def test_an_empty_registry_lists_nothing() -> None:
    assert (
        InMemoryServedModelListing(InMemoryServedModelRepository()).page(
            after=None, limit=5, state=None
        )
        == ()
    )
