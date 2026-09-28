from dataclasses import replace

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.in_memory.verdict_memo import InMemoryVerdictMemo
from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary
from emblema.evaluation.application.use_cases.view_campaign import ViewCampaign, ViewCampaignQuery
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.exceptions import CampaignNotFoundError
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.verdict_memo import VerdictMemo
from tests.evaluation.support import (
    BUDGETS,
    CAMPAIGN,
    CONTENDER,
    CONTROL,
    SELECTED_BY,
    closed_reading,
    ran_reading,
    reading,
    selection,
    store,
)


def view(*stored: CampaignReading, memo: VerdictMemo | None = None) -> ViewCampaign:
    memo = InMemoryVerdictMemo(capacity=4) if memo is None else memo
    repository = InMemoryEvaluationCampaignRepository()
    for read in stored:
        store(repository, read)
    return ViewCampaign(repository, memo)


def test_a_finished_comparison_is_shown_with_its_curves_and_its_verdict() -> None:
    closed = closed_reading()

    shown = view(closed)(ViewCampaignQuery(campaign=CAMPAIGN))

    assert shown.summary == CampaignSummary.of(CampaignOverview.of(closed.campaign))
    assert shown.summary.finished
    assert shown.summary.cells_recorded == shown.summary.cells_planned == 8
    assert [(c.candidate, c.kind) for c in shown.curves] == [
        (CONTROL, CandidateKind.NEURAL),
        (CONTENDER, CandidateKind.NEURAL),
    ]
    control, contender = shown.curves
    assert [p.budget for p in control.points] == ["50", "200"]
    assert [p.repeats for p in control.points] == [2, 2]
    assert control.points[0].error == pytest.approx(closed.error_of(CONTROL, BUDGETS[0]))
    assert contender.points[1].error == pytest.approx(closed.error_of(CONTENDER, BUDGETS[1]))
    assert shown.verdict is not None
    assert shown.verdict.sentence == closed.verdict().sentence()
    assert shown.verdict.control == CONTROL
    assert shown.verdict.read_on is RunPurpose.TUNING
    assert shown.verdict.endpoint.candidate == CONTENDER
    assert shown.verdict.endpoint.budget == "200"
    assert shown.verdict.endpoint.interval_level == 0.95
    assert len(shown.verdict.secondary) == 1


def test_a_running_campaign_shows_its_curves_so_far_and_no_verdict() -> None:
    partial = reading()
    for produced in ran_reading().results[:3]:
        partial = partial.record(produced)

    shown = view(partial)(ViewCampaignQuery(campaign=CAMPAIGN))

    assert not shown.summary.finished
    assert shown.summary.cells_recorded == 3
    assert shown.verdict is None
    control, contender = shown.curves
    assert [p.repeats for p in control.points] == [2, 1]
    assert [p.repeats for p in contender.points] == [0, 0]
    assert contender.points[0].error is None


def test_a_selection_shows_no_verdict_because_it_concludes_nothing() -> None:
    shown = view(selection())(ViewCampaignQuery(campaign=SELECTED_BY))

    assert shown.summary.purpose is RunPurpose.SELECTION
    assert shown.verdict is None


def test_the_verdict_of_a_closed_campaign_is_kept_under_its_revision_once_read() -> None:
    closed = closed_reading()
    memo = InMemoryVerdictMemo(capacity=4)
    use_case = view(closed, memo=memo)

    first = use_case(ViewCampaignQuery(campaign=CAMPAIGN))
    second = use_case(ViewCampaignQuery(campaign=CAMPAIGN))

    assert first.verdict == second.verdict
    assert memo.recall(CAMPAIGN, closed.campaign.revision) == closed.verdict()


def test_a_verdict_kept_is_answered_without_reading_it_again() -> None:
    closed = closed_reading()
    memo = InMemoryVerdictMemo(capacity=4)
    # Told apart from what reading it again would give: the real verdict has one secondary.
    kept = replace(closed.verdict(), secondary=())
    memo.keep(CAMPAIGN, closed.campaign.revision, kept)

    shown = view(closed, memo=memo)(ViewCampaignQuery(campaign=CAMPAIGN))

    assert shown.verdict is not None
    assert shown.verdict.secondary == ()


def test_a_campaign_nobody_stored_is_not_found() -> None:
    with pytest.raises(CampaignNotFoundError):
        view()(ViewCampaignQuery(campaign=CAMPAIGN))
