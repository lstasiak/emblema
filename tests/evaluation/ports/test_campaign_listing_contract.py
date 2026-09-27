"""Contract of the CampaignListing port, run against every adapter.

Both adapters order and cut the same way, so a page read from the database is the page read
from memory; the database adapter needs the metadata database and is marked ``integration``.
"""

from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import Engine

from emblema.evaluation.adapters.in_memory.campaign_listing import InMemoryCampaignListing
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.persistence.campaign_listing import SqlAlchemyCampaignListing
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.exceptions import CampaignNotFoundError
from emblema.evaluation.ports.campaign_listing import CampaignListing
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import OPENED_AT, closed_reading, reading, store
from tests.support.database import clear_evaluation, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]


def designed(number: int, *, hours: int) -> CampaignReading:
    return reading(
        campaign_id=CampaignId(UUID(int=number)),
        opened_at=UtcDateTime(OPENED_AT.value + timedelta(hours=hours)),
    )


# The closed campaign was designed first; two later ones were designed at the same instant.
CLOSED = closed_reading()
CAMPAIGNS = (CLOSED, designed(5, hours=1), designed(6, hours=1))
NEWEST_FIRST = tuple(CampaignOverview.of(c.campaign) for c in (CAMPAIGNS[1], CAMPAIGNS[2], CLOSED))
GRID = tuple(
    sorted(CLOSED.results, key=lambda r: (str(r.cell.candidate), r.cell.budget.text(), r.cell.seed))
)


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def listing(request: pytest.FixtureRequest) -> CampaignListing:
    if request.param == "in_memory":
        repository = InMemoryEvaluationCampaignRepository()
        for stored in CAMPAIGNS:
            store(repository, stored)
        return InMemoryCampaignListing(repository)
    engine: Engine = request.getfixturevalue("database")
    clear_evaluation(engine)
    for stored in CAMPAIGNS:
        store(SqlAlchemyEvaluationCampaignRepository(engine), stored)
    return SqlAlchemyCampaignListing(engine)


def test_campaigns_are_listed_newest_first_identity_breaking_a_tie(
    listing: CampaignListing,
) -> None:
    assert listing.campaigns(after=None, limit=10) == NEWEST_FIRST


def test_a_page_of_campaigns_continues_after_the_one_named(listing: CampaignListing) -> None:
    first = listing.campaigns(after=None, limit=2)

    second = listing.campaigns(after=first[-1].position, limit=2)

    assert first == NEWEST_FIRST[:2]
    assert second == NEWEST_FIRST[2:]


def test_a_page_continues_after_a_position_no_stored_campaign_holds(
    listing: CampaignListing,
) -> None:
    # Half an hour before the two later campaigns, under an identity nobody holds.
    between = CampaignPosition(
        opened_at=UtcDateTime(CAMPAIGNS[1].campaign.opened_at.value - timedelta(minutes=30)),
        campaign_id=CampaignId(UUID(int=99)),
    )

    assert listing.campaigns(after=between, limit=10) == NEWEST_FIRST[2:]


def test_a_listed_campaign_counts_its_cells_without_carrying_them(
    listing: CampaignListing,
) -> None:
    (closed,) = [c for c in listing.campaigns(after=None, limit=10) if c.is_finished]

    assert closed.cells_recorded == len(CLOSED.results)


def test_the_results_of_a_campaign_are_listed_by_candidate_budget_and_seed(
    listing: CampaignListing,
) -> None:
    assert listing.results(CLOSED.campaign.campaign_id, after=None, limit=10) == GRID


def test_a_page_of_results_continues_after_the_cell_named(listing: CampaignListing) -> None:
    first = listing.results(CLOSED.campaign.campaign_id, after=None, limit=3)

    second = listing.results(CLOSED.campaign.campaign_id, after=first[-1].cell, limit=3)
    rest = listing.results(CLOSED.campaign.campaign_id, after=second[-1].cell, limit=3)

    assert first + second + rest == GRID
    assert listing.results(CLOSED.campaign.campaign_id, after=rest[-1].cell, limit=3) == ()


def test_a_campaign_that_has_not_run_lists_no_result(listing: CampaignListing) -> None:
    assert listing.results(CAMPAIGNS[1].campaign.campaign_id, after=None, limit=10) == ()


def test_the_results_of_a_campaign_nobody_stored_are_refused(listing: CampaignListing) -> None:
    with pytest.raises(CampaignNotFoundError):
        listing.results(CampaignId(UUID(int=99)), after=None, limit=10)
