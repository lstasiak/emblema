"""Contract of the EvaluationCampaignRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. Both store a whole state and give it back whole — the
design as it was declared, every cell that has run, and the errors each cell made per unit,
because those are what the comparison is recomputed from. Both also refuse a write over a state
the writer never read, which is what keeps two workers of one grid from deleting each other's
cells; the database adapter is where that has to hold under real transactions.
"""

import pytest
from sqlalchemy import Engine

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignNotFoundError,
)
from emblema.evaluation.ports.evaluation_campaign_repository import (
    EvaluationCampaignRepository,
)
from tests.evaluation.support import (
    CAMPAIGN,
    CONTENDER,
    artifact,
    campaign,
    closed_campaign,
    result,
)
from tests.support.database import clear_evaluation, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]
ERRORS = (3.0, 4.0, 5.0)


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def campaigns(request: pytest.FixtureRequest) -> EvaluationCampaignRepository:
    if request.param == "in_memory":
        return InMemoryEvaluationCampaignRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_evaluation(engine)
    return SqlAlchemyEvaluationCampaignRepository(engine)


def test_a_campaign_with_nothing_run_comes_back_as_it_was_declared(
    campaigns: EvaluationCampaignRepository,
) -> None:
    declared = campaign()

    campaigns.save(declared, seen=declared.revision)

    assert campaigns.get(CAMPAIGN) == declared


def test_the_cells_that_ran_come_back_with_their_errors_and_what_they_kept(
    campaigns: EvaluationCampaignRepository,
) -> None:
    cell = campaign().design.cells()[0]
    advanced = campaign().record(
        result(cell.candidate, cell.budget, cell.seed, ERRORS, artifact=artifact("kept"))
    )

    campaigns.save(advanced, seen=0)

    stored = campaigns.get(CAMPAIGN)
    assert stored.results == advanced.results
    assert stored.pending() == advanced.pending()


def test_a_finished_campaign_comes_back_finished(
    campaigns: EvaluationCampaignRepository,
) -> None:
    campaigns.save(closed_campaign(), seen=0)

    stored = campaigns.get(CAMPAIGN)
    assert stored.is_finished
    assert stored.verdict().endpoint.candidate == CONTENDER


def test_saving_again_replaces_the_state_rather_than_adding_to_it(
    campaigns: EvaluationCampaignRepository,
) -> None:
    cell = campaign().design.cells()[0]
    campaigns.save(campaign(), seen=0)

    campaigns.save(
        campaign().record(result(cell.candidate, cell.budget, cell.seed, ERRORS)), seen=0
    )

    assert len(campaigns.get(CAMPAIGN).results) == 1


def test_a_campaign_nobody_stored_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    with pytest.raises(CampaignNotFoundError):
        campaigns.get(CAMPAIGN)


def test_a_write_over_a_state_the_writer_never_read_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    # Two workers read the same empty grid and each ran a cell of it. The first writes; the
    # second is holding a campaign that knows nothing of the first's cell, and writing it whole
    # would delete that cell.
    read = campaign()
    first, second = read.design.cells()[0], read.design.cells()[1]
    campaigns.save(read, seen=read.revision)
    campaigns.save(
        read.record(result(first.candidate, first.budget, first.seed, ERRORS)), seen=read.revision
    )

    with pytest.raises(CampaignChangedElsewhereError):
        campaigns.save(
            read.record(result(second.candidate, second.budget, second.seed, ERRORS)),
            seen=read.revision,
        )

    assert [stored.cell for stored in campaigns.get(CAMPAIGN).results] == [first]


def test_a_writer_that_reads_again_records_beside_what_it_lost_to(
    campaigns: EvaluationCampaignRepository,
) -> None:
    read = campaign()
    first, second = read.design.cells()[0], read.design.cells()[1]
    campaigns.save(read, seen=read.revision)
    campaigns.save(
        read.record(result(first.candidate, first.budget, first.seed, ERRORS)), seen=read.revision
    )

    again = campaigns.get(CAMPAIGN)
    campaigns.save(
        again.record(result(second.candidate, second.budget, second.seed, ERRORS)),
        seen=again.revision,
    )

    assert [stored.cell for stored in campaigns.get(CAMPAIGN).results] == [first, second]
