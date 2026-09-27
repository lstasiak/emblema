"""A campaign read with its results: the verdict, the figures and what the results must be."""

from dataclasses import replace

import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
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
    ran_reading,
    reading,
    result,
)

ERRORS = (3.0, 4.0, 5.0)
AT_200 = LabelBudget.of(200)


def test_a_reading_holds_exactly_one_result_per_recorded_cell() -> None:
    first, second = campaign().design.cells()[:2]
    produced = result(first.candidate, first.budget, first.seed, ERRORS)
    other = result(second.candidate, second.budget, second.seed, ERRORS)

    with pytest.raises(UnknownCampaignCellError, match="has not recorded"):
        CampaignReading(campaign=campaign(), results=(produced,))
    with pytest.raises(IncompleteCampaignError, match="results are missing"):
        CampaignReading(campaign=campaign().record(first), results=())
    with pytest.raises(CampaignCellAlreadyRecordedError, match="two results"):
        CampaignReading(campaign=campaign().record(first), results=(produced, produced))
    with pytest.raises(UnknownCampaignCellError):
        CampaignReading(campaign=campaign().record(first), results=(other,))


def test_recording_a_result_records_its_cell_and_keeps_the_result() -> None:
    first = campaign().design.cells()[0]
    produced = result(first.candidate, first.budget, first.seed, ERRORS)

    advanced = reading().record(produced)

    assert advanced.campaign.recorded == (first,)
    assert advanced.results == (produced,)
    assert advanced.get_result(first) == produced


def test_a_result_of_a_cell_nobody_ran_is_refused() -> None:
    with pytest.raises(UnknownCampaignCellError, match="has not run"):
        reading().get_result(campaign().design.cells()[0])


def test_a_reading_of_a_finished_campaign_records_no_further_cell() -> None:
    finished = ran_reading().complete(OPENED_AT)
    first = finished.campaign.design.cells()[0]

    with pytest.raises(CampaignClosedError):
        finished.record(result(first.candidate, first.budget, first.seed, ERRORS))


def test_a_verdict_is_refused_until_the_campaign_has_finished() -> None:
    with pytest.raises(CampaignNotCompletedError):
        ran_reading().verdict()


def test_the_verdict_reads_the_endpoint_apart_from_the_secondary_family() -> None:
    verdict = ran_reading().complete(OPENED_AT).verdict()

    assert verdict.endpoint.candidate == CONTENDER
    assert verdict.endpoint.budget == AT_200
    assert [(c.candidate, c.budget) for c in verdict.secondary] == [(CONTENDER, LabelBudget.of(50))]


def test_the_endpoint_is_confirmed_when_the_contender_errs_less_on_every_unit() -> None:
    verdict = ran_reading().complete(OPENED_AT).verdict()

    assert verdict.endpoint.difference.reduction > 0.0
    assert verdict.endpoint.difference.interval.above_zero
    assert str(verdict.endpoint.verdict) == "confirmed"


def test_a_contender_that_errs_as_the_control_does_is_not_told_apart() -> None:
    same = ran_reading(errors={CONTROL: ERRORS, CONTENDER: ERRORS})

    verdict = same.complete(OPENED_AT).verdict()

    assert verdict.endpoint.difference.reduction == 0.0
    assert str(verdict.endpoint.verdict) == "indistinguishable"


def test_the_repeats_of_a_cell_are_pooled_rather_than_averaged() -> None:
    whole = ran_reading()

    assert len(whole.results_of(CONTENDER, AT_200)) == 2
    assert whole.error_of(CONTENDER, AT_200) == pytest.approx(
        whole.results_of(CONTENDER, AT_200)[0].rmse
    )


def test_a_candidate_with_nothing_run_has_no_score() -> None:
    with pytest.raises(UnknownCandidateError):
        reading().error_of(CONTENDER, AT_200)
    with pytest.raises(UnknownCandidateError):
        reading().mean_squared_error_of(CONTENDER, AT_200)


def test_the_kept_artifact_is_the_one_from_the_cell_the_design_named() -> None:
    assert ran_reading(artifact("kept")).artifact_of(CONTENDER) == artifact("kept")


def test_a_campaign_that_has_kept_nothing_names_no_artifact() -> None:
    assert reading().artifact_of(CONTROL) is None


def test_the_sentence_names_the_endpoint_its_budget_and_what_was_found() -> None:
    sentence = ran_reading().complete(OPENED_AT).verdict().sentence()

    assert "200 labels" in sentence
    assert str(CONTENDER) in sentence
    assert sentence.startswith("Confirmed on the registered endpoint")
    assert sentence.endswith("Validation side; preliminary.")


def test_a_pairing_the_campaign_never_compared_is_refused() -> None:
    verdict = ran_reading().complete(OPENED_AT).verdict()

    with pytest.raises(UnknownCandidateError, match="no comparison of"):
        verdict.get_comparison(CandidateRef("absent"), AT_200)


def test_a_budget_the_campaign_never_ran_is_refused_by_the_verdict() -> None:
    verdict = ran_reading().complete(OPENED_AT).verdict()

    with pytest.raises(UnknownCandidateError, match="at a budget of all"):
        verdict.get_comparison(CONTENDER, LabelBudget.everything())


def test_closing_a_reading_closes_its_campaign_and_keeps_its_results() -> None:
    whole = ran_reading()

    closed = whole.complete(OPENED_AT)

    assert closed.campaign == whole.campaign.complete(OPENED_AT)
    assert closed.results == whole.results
    assert replace(closed, campaign=whole.campaign) == whole
