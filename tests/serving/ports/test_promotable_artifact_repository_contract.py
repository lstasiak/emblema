"""Contract of the PromotableArtifactRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. What both owe is the projection back whole — its
scores in the order they were announced, the budget that counts no window included — found by
the checksum a promotion names, and one artifact per origin however often it was announced.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine

from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.adapters.persistence.promotable_artifact_repository import (
    SqlAlchemyPromotableArtifactRepository,
)
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.serving.support import KEPT, OTHER_CAMPAIGN, origin, promotable, score
from tests.support.database import clear_serving, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]
EARLIER = UtcDateTime(datetime(2026, 8, 1, tzinfo=UTC))


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def promotables(request: pytest.FixtureRequest) -> PromotableArtifactRepository:
    if request.param == "in_memory":
        return InMemoryPromotableArtifactRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_serving(engine)
    return SqlAlchemyPromotableArtifactRepository(engine)


def test_a_kept_artifact_is_found_by_its_checksum_as_it_was_stored(
    promotables: PromotableArtifactRepository,
) -> None:
    stored = promotable()

    promotables.save(stored)

    assert promotables.find_by_checksum(KEPT.checksum) == (stored,)


def test_an_artifact_announced_again_replaces_what_the_first_announcement_left(
    promotables: PromotableArtifactRepository,
) -> None:
    promotables.save(promotable())
    again = promotable(standing=CandidateStanding.NOT_ESTABLISHED, scores=(score(),))

    promotables.save(again)

    assert promotables.find_by_checksum(KEPT.checksum) == (again,)


def test_a_checksum_no_campaign_kept_finds_nothing(
    promotables: PromotableArtifactRepository,
) -> None:
    promotables.save(promotable())

    assert promotables.find_by_checksum(Checksum.of_bytes(b"never kept")) == ()


def test_one_artifact_kept_by_several_campaigns_is_found_under_each_in_order_of_finishing(
    promotables: PromotableArtifactRepository,
) -> None:
    later = promotable()
    earlier = promotable(origin=origin(campaign=OTHER_CAMPAIGN), completed_at=EARLIER)
    promotables.save(later)
    promotables.save(earlier)

    assert promotables.find_by_checksum(KEPT.checksum) == (earlier, later)


def test_competitors_of_one_campaign_are_ordered_by_code_point_not_by_collation(
    promotables: PromotableArtifactRepository,
) -> None:
    # Upper case sorts before lower case by code point and after it under most collations.
    lower = promotable(origin=origin(candidate=CandidateRef("a_trees")))
    upper = promotable(origin=origin(candidate=CandidateRef("B_trees")))
    promotables.save(lower)
    promotables.save(upper)

    assert promotables.find_by_checksum(KEPT.checksum) == (upper, lower)
