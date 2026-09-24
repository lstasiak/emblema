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
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    OPENED_AT,
    artifact,
    campaign,
    result,
)

ERRORS = (3.0, 4.0, 5.0)


def run_everything(errors: dict[CandidateRef, tuple[float, ...]] | None = None):
    """A campaign with every cell of its grid recorded, the contender erring less by default."""
    scored = errors or {CONTROL: (6.0, 8.0, 10.0), CONTENDER: ERRORS}
    whole = campaign()
    for cell in whole.design.cells():
        whole = whole.record(result(cell.candidate, cell.budget, cell.seed, scored[cell.candidate]))
    return whole


def test_a_fresh_campaign_has_its_whole_grid_pending() -> None:
    fresh = campaign()

    assert fresh.pending() == fresh.design.cells()
    assert not fresh.is_complete
    assert not fresh.is_finished


def test_recording_a_cell_takes_it_off_the_pending_list() -> None:
    first = campaign().design.cells()[0]

    advanced = campaign().record(result(first.candidate, first.budget, first.seed, ERRORS))

    assert first not in advanced.pending()
    assert len(advanced.pending()) == len(campaign().pending()) - 1


def test_a_result_for_a_cell_the_grid_does_not_hold_is_refused() -> None:
    with pytest.raises(UnknownCampaignCellError):
        campaign().record(result(CandidateRef("absent"), LabelBudget.of(50), 1, ERRORS))


def test_a_cell_recorded_twice_is_refused() -> None:
    first = campaign().design.cells()[0]
    once = campaign().record(result(first.candidate, first.budget, first.seed, ERRORS))

    with pytest.raises(CampaignCellAlreadyRecordedError):
        once.record(result(first.candidate, first.budget, first.seed, ERRORS))


def test_a_campaign_with_cells_left_to_run_refuses_to_finish() -> None:
    with pytest.raises(IncompleteCampaignError):
        campaign().complete(OPENED_AT)


def test_a_campaign_that_has_finished_records_no_further_cell() -> None:
    finished = run_everything().complete(OPENED_AT)
    first = finished.design.cells()[0]

    with pytest.raises(CampaignClosedError):
        finished.record(result(first.candidate, first.budget, first.seed, ERRORS))


def test_a_campaign_that_has_finished_does_not_finish_twice() -> None:
    finished = run_everything().complete(OPENED_AT)

    with pytest.raises(CampaignClosedError):
        finished.complete(OPENED_AT)


def test_a_verdict_is_refused_until_the_campaign_has_finished() -> None:
    with pytest.raises(CampaignNotCompletedError):
        run_everything().verdict()


def test_a_campaign_built_finished_with_cells_missing_is_refused() -> None:
    with pytest.raises(IncompleteCampaignError):
        replace(campaign(), completed_at=OPENED_AT)


def test_the_verdict_reads_the_endpoint_apart_from_the_secondary_family() -> None:
    verdict = run_everything().complete(OPENED_AT).verdict()

    assert verdict.endpoint.candidate == CONTENDER
    assert verdict.endpoint.budget == LabelBudget.of(200)
    assert [(c.candidate, c.budget) for c in verdict.secondary] == [(CONTENDER, LabelBudget.of(50))]


def test_the_endpoint_is_confirmed_when_the_contender_errs_less_on_every_unit() -> None:
    verdict = run_everything().complete(OPENED_AT).verdict()

    assert verdict.endpoint.difference.reduction > 0.0
    assert verdict.endpoint.difference.interval.above_zero
    assert str(verdict.endpoint.verdict) == "confirmed"


def test_a_contender_that_errs_as_the_control_does_is_not_told_apart() -> None:
    same = run_everything({CONTROL: ERRORS, CONTENDER: ERRORS})

    verdict = same.complete(OPENED_AT).verdict()

    assert verdict.endpoint.difference.reduction == 0.0
    assert str(verdict.endpoint.verdict) == "indistinguishable"


def test_the_repeats_of_a_cell_are_pooled_rather_than_averaged() -> None:
    whole = run_everything()

    assert len(whole.results_of(CONTENDER, LabelBudget.of(200))) == 2
    assert whole.rmse_of(CONTENDER, LabelBudget.of(200)) == pytest.approx(
        whole.results_of(CONTENDER, LabelBudget.of(200))[0].rmse
    )


def test_a_candidate_with_nothing_run_has_no_score() -> None:
    with pytest.raises(UnknownCandidateError):
        campaign().rmse_of(CONTENDER, LabelBudget.of(200))


def test_the_kept_artifact_is_the_one_from_the_cell_the_design_named() -> None:
    whole = campaign()
    for cell in whole.design.cells():
        kept = "kept" if whole.design.retains(cell) else None
        whole = whole.record(
            result(
                cell.candidate,
                cell.budget,
                cell.seed,
                ERRORS,
                artifact=None if kept is None else artifact(kept),
            )
        )

    assert whole.artifact_of(CONTENDER) == artifact("kept")


def test_the_sentence_names_the_endpoint_its_budget_and_what_was_found() -> None:
    sentence = run_everything().complete(OPENED_AT).verdict().sentence()

    assert "200 labels" in sentence
    assert str(CONTENDER) in sentence
    assert "confirmed" in sentence


def test_a_cell_is_the_coordinates_and_reads_as_them() -> None:
    stated = CampaignCell(candidate=CONTENDER, budget=LabelBudget.everything(), seed=3)

    assert str(stated) == "full_fine_tuning at all under seed 3"


def test_a_pairing_the_campaign_never_compared_is_refused() -> None:
    verdict = run_everything().complete(OPENED_AT).verdict()

    with pytest.raises(UnknownCandidateError, match="no comparison of"):
        verdict.get_comparison(CandidateRef("absent"), LabelBudget.of(200))


def test_a_budget_the_campaign_never_ran_is_refused_by_the_verdict() -> None:
    verdict = run_everything().complete(OPENED_AT).verdict()

    with pytest.raises(UnknownCandidateError, match="at a budget of all"):
        verdict.get_comparison(CONTENDER, LabelBudget.everything())


def test_a_campaign_that_has_kept_nothing_names_no_artifact() -> None:
    assert campaign().artifact_of(CONTROL) is None


def test_a_campaign_nobody_has_run_stands_at_no_revision() -> None:
    assert campaign().revision == 0


def test_every_change_a_campaign_can_undergo_moves_its_revision_by_one() -> None:
    # The two are the whole of what can change: who competes, over what and how the result is
    # read are settled before anything runs. That is what lets the count of changes be counted
    # off the state rather than carried in a field beside it.
    first = campaign().design.cells()[0]

    recorded = campaign().record(result(first.candidate, first.budget, first.seed, ERRORS))
    whole = run_everything()
    finished = whole.complete(OPENED_AT)

    assert recorded.revision == campaign().revision + 1
    assert finished.revision == whole.revision + 1


def test_a_campaign_read_back_stands_where_it_stood() -> None:
    whole = run_everything()

    assert whole.revision == len(whole.design.cells())
