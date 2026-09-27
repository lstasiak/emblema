from dataclasses import replace

import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.exceptions import (
    CampaignCellAlreadyRecordedError,
    CampaignClosedError,
    CampaignNotCompletedError,
    IncompleteCampaignError,
    UnknownCampaignCellError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from tests.evaluation.support import CONTROL, OPENED_AT, campaign, ran_reading


def whole_grid() -> tuple[CampaignCell, ...]:
    return campaign().design.cells()


def ran() -> tuple[CampaignCell, ...]:
    """The campaign of ``ran_reading``, every cell recorded, still open."""
    return ran_reading().campaign.recorded


def test_a_fresh_campaign_has_its_whole_grid_pending() -> None:
    fresh = campaign()

    assert fresh.pending() == fresh.design.cells()
    assert not fresh.is_complete
    assert not fresh.is_finished


def test_recording_a_cell_takes_it_off_the_pending_list() -> None:
    first = whole_grid()[0]

    advanced = campaign().record(first)

    assert first not in advanced.pending()
    assert advanced.recorded == (first,)
    assert len(advanced.pending()) == len(campaign().pending()) - 1


def test_a_cell_the_grid_does_not_hold_is_refused() -> None:
    with pytest.raises(UnknownCampaignCellError):
        campaign().record(
            CampaignCell(candidate=CandidateRef("absent"), budget=LabelBudget.of(50), seed=1)
        )


def test_a_cell_recorded_twice_is_refused() -> None:
    first = whole_grid()[0]
    once = campaign().record(first)

    with pytest.raises(CampaignCellAlreadyRecordedError):
        once.record(first)


def test_a_campaign_with_cells_left_to_run_refuses_to_finish() -> None:
    with pytest.raises(IncompleteCampaignError):
        campaign().complete(OPENED_AT)


def test_a_campaign_that_has_finished_records_no_further_cell() -> None:
    finished = ran_reading().campaign.complete(OPENED_AT)

    with pytest.raises(CampaignClosedError):
        finished.record(whole_grid()[0])


def test_a_campaign_that_has_finished_does_not_finish_twice() -> None:
    finished = ran_reading().campaign.complete(OPENED_AT)

    with pytest.raises(CampaignClosedError):
        finished.complete(OPENED_AT)


def test_a_campaign_built_finished_with_cells_missing_is_refused() -> None:
    with pytest.raises(IncompleteCampaignError):
        replace(campaign(), completed_at=OPENED_AT)


def test_a_closed_campaign_says_when_it_closed() -> None:
    assert ran_reading().campaign.complete(OPENED_AT).completion() == OPENED_AT


def test_a_campaign_still_running_has_no_moment_of_closing() -> None:
    with pytest.raises(CampaignNotCompletedError):
        ran_reading().campaign.completion()


def test_a_cell_is_the_coordinates_and_reads_as_them() -> None:
    stated = CampaignCell(candidate=CONTROL, budget=LabelBudget.everything(), seed=3)

    assert str(stated) == "from_scratch at all under seed 3"


def test_a_campaign_nobody_has_run_stands_at_no_revision() -> None:
    assert campaign().revision == 0


def test_every_change_a_campaign_can_undergo_moves_its_revision_by_one() -> None:
    # The two are the whole of what can change: who competes, over what and how the result is
    # read are settled before anything runs. That is what lets the count of changes be counted
    # off the state rather than carried in a field beside it.
    recorded = campaign().record(whole_grid()[0])
    whole = ran_reading().campaign
    finished = whole.complete(OPENED_AT)

    assert recorded.revision == campaign().revision + 1
    assert finished.revision == whole.revision + 1


def test_a_campaign_read_back_stands_where_it_stood() -> None:
    whole = ran_reading().campaign

    assert whole.revision == len(whole.design.cells())
    assert whole.cells_recorded == len(ran())


def test_a_cell_asks_its_candidate_what_the_design_declared_under_the_campaigns_rules() -> None:
    grid = campaign()
    kept = next(cell for cell in grid.design.cells() if grid.design.retains(cell))

    asked = grid.evaluation_of(kept)

    assert asked.task == grid.task
    assert asked.purpose == grid.purpose
    assert asked.retain is True
    assert asked.declared == grid.design.get_candidate(kept.candidate)


def test_a_cell_outside_the_grid_cannot_be_asked_of_anyone() -> None:
    with pytest.raises(UnknownCampaignCellError):
        campaign().evaluation_of(
            CampaignCell(candidate=CONTROL, budget=LabelBudget.of(50), seed=99)
        )


def test_results_are_accepted_only_one_per_recorded_cell() -> None:
    first, second = whole_grid()[:2]
    recorded = campaign().record(first)

    recorded.accept_results([first])
    with pytest.raises(UnknownCampaignCellError, match="has not recorded"):
        recorded.accept_results([first, second])
    with pytest.raises(CampaignCellAlreadyRecordedError, match="two results"):
        recorded.accept_results([first, first])
    with pytest.raises(IncompleteCampaignError, match="results are missing"):
        recorded.accept_results([])
