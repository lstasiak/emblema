"""Contract of the EvaluationCampaignRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. Both give a campaign back as it stands and its reading
whole — the design as it was declared, every cell that has run, and the errors and answers each
cell recorded, because those are what the comparison is recomputed from — however many writes
it took to record it. Both also refuse a write over a state the writer never read, which is what
keeps two workers of one grid from losing each other's cells; the database adapter is where that
has to hold under real transactions.
"""

from dataclasses import replace

import pytest
from sqlalchemy import Engine

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignNotFoundError,
    IncompleteCampaignError,
    UnknownCampaignCellError,
)
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from emblema.evaluation.ports.evaluation_campaign_repository import (
    EvaluationCampaignRepository,
)
from tests.evaluation.support import (
    CAMPAIGN,
    CONTENDER,
    RULES,
    answered,
    artifact,
    campaign,
    closed_reading,
    design,
    result,
    store,
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
    assert campaigns.read(CAMPAIGN) == CampaignReading(campaign=declared, results=())


def test_a_cell_recorded_comes_back_in_the_campaign_and_with_its_result(
    campaigns: EvaluationCampaignRepository,
) -> None:
    declared = campaign()
    cell = declared.design.cells()[0]
    produced = result(cell.candidate, cell.budget, cell.seed, ERRORS, artifact=artifact("kept"))
    campaigns.save(declared, seen=0)

    campaigns.record(declared.record(cell), produced, seen=declared.revision)

    stored = campaigns.get(CAMPAIGN)
    assert stored.recorded == (cell,)
    assert stored.pending() == declared.pending()[1:]
    assert campaigns.get_result(CAMPAIGN, cell) == produced
    assert campaigns.read(CAMPAIGN).results == (produced,)


def test_a_campaign_read_by_area_comes_back_with_its_rules_and_every_answer(
    campaigns: EvaluationCampaignRepository,
) -> None:
    rules = replace(
        RULES, threshold=ThresholdKind.ABSOLUTE, minimum_reduction=0.02, floor_part=0.01
    )
    declared = campaign(design=replace(design(), measure=ErrorMeasure.AUROC_SHORTFALL, rules=rules))
    cell = declared.design.cells()[0]
    produced = answered(cell.candidate, cell.budget, cell.seed, (0.9, 0.2, 0.7, 0.4, 0.1, 0.3))
    campaigns.save(declared, seen=0)

    campaigns.record(declared.record(cell), produced, seen=declared.revision)

    assert campaigns.read(CAMPAIGN) == CampaignReading(
        campaign=declared.record(cell), results=(produced,)
    )


def test_cells_recorded_over_several_writes_come_back_whole_in_key_order(
    campaigns: EvaluationCampaignRepository,
) -> None:
    whole = closed_reading(artifact("kept"))

    store(campaigns, whole)

    read = campaigns.read(CAMPAIGN)
    assert read.campaign.is_finished
    assert read.campaign.revision == whole.campaign.revision
    assert set(read.campaign.recorded) == set(whole.campaign.recorded)
    assert set(read.results) == set(whole.results)
    assert read.verdict() == whole.verdict()


def test_a_finished_campaign_comes_back_finished(
    campaigns: EvaluationCampaignRepository,
) -> None:
    store(campaigns, closed_reading())

    stored = campaigns.get(CAMPAIGN)
    assert stored.is_finished
    assert campaigns.read(CAMPAIGN).verdict().endpoint.candidate == CONTENDER


def test_a_campaign_nobody_stored_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    with pytest.raises(CampaignNotFoundError):
        campaigns.get(CAMPAIGN)
    with pytest.raises(CampaignNotFoundError):
        campaigns.read(CAMPAIGN)
    with pytest.raises(CampaignNotFoundError):
        campaigns.get_result(CAMPAIGN, campaign().design.cells()[0])


def test_a_cell_nobody_recorded_has_no_result(
    campaigns: EvaluationCampaignRepository,
) -> None:
    campaigns.save(campaign(), seen=0)

    with pytest.raises(UnknownCampaignCellError):
        campaigns.get_result(CAMPAIGN, campaign().design.cells()[0])


def test_a_result_for_a_cell_the_campaign_does_not_record_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    declared = campaign()
    first, second = declared.design.cells()[:2]
    campaigns.save(declared, seen=0)

    with pytest.raises(UnknownCampaignCellError):
        campaigns.record(
            declared.record(first),
            result(second.candidate, second.budget, second.seed, ERRORS),
            seen=declared.revision,
        )


def test_a_campaign_recording_a_cell_whose_result_was_never_stored_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    declared = campaign()
    cell = declared.design.cells()[0]
    campaigns.save(declared, seen=0)

    with pytest.raises(IncompleteCampaignError):
        campaigns.save(declared.record(cell), seen=declared.revision)

    assert campaigns.get(CAMPAIGN) == declared


def test_a_write_over_a_state_the_writer_never_read_is_refused(
    campaigns: EvaluationCampaignRepository,
) -> None:
    # Two workers read the same empty grid and each ran a cell of it. The first writes; the
    # second is holding a campaign that knows nothing of the first's cell.
    read = campaign()
    first, second = read.design.cells()[0], read.design.cells()[1]
    campaigns.save(read, seen=read.revision)
    campaigns.record(
        read.record(first),
        result(first.candidate, first.budget, first.seed, ERRORS),
        seen=read.revision,
    )

    with pytest.raises(CampaignChangedElsewhereError):
        campaigns.record(
            read.record(second),
            result(second.candidate, second.budget, second.seed, ERRORS),
            seen=read.revision,
        )

    assert campaigns.get(CAMPAIGN).recorded == (first,)


def test_a_writer_that_reads_again_records_beside_what_it_lost_to(
    campaigns: EvaluationCampaignRepository,
) -> None:
    read = campaign()
    first, second = read.design.cells()[0], read.design.cells()[1]
    campaigns.save(read, seen=read.revision)
    campaigns.record(
        read.record(first),
        result(first.candidate, first.budget, first.seed, ERRORS),
        seen=read.revision,
    )

    again = campaigns.get(CAMPAIGN)
    campaigns.record(
        again.record(second),
        result(second.candidate, second.budget, second.seed, ERRORS),
        seen=again.revision,
    )

    assert set(campaigns.get(CAMPAIGN).recorded) == {first, second}
    assert {r.cell for r in campaigns.read(CAMPAIGN).results} == {first, second}
