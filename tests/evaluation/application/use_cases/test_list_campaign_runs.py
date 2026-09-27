from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.campaign_listing import InMemoryCampaignListing
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.read_models.campaign_run import CampaignRun
from emblema.evaluation.application.use_cases.list_campaign_runs import (
    ListCampaignRuns,
    ListCampaignRunsQuery,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.exceptions import CampaignNotFoundError
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from tests.evaluation.support import CAMPAIGN, artifact, closed_campaign

KEPT = artifact("kept")


def listing() -> ListCampaignRuns:
    repository = InMemoryEvaluationCampaignRepository()
    repository.save(closed_campaign(KEPT), seen=0)
    return ListCampaignRuns(InMemoryCampaignListing(repository))


def test_the_runs_of_a_campaign_are_listed_whole_when_the_page_is_large_enough() -> None:
    page = listing()(ListCampaignRunsQuery(campaign=CAMPAIGN, limit=10))

    assert len(page.items) == 8
    assert page.next_cursor is None
    assert [(str(r.candidate), r.budget, r.seed) for r in page.items[:3]] == [
        ("from_scratch", "200", 1),
        ("from_scratch", "200", 2),
        ("from_scratch", "50", 1),
    ]
    kept = [r for r in page.items if r.artifact is not None]
    assert [(str(r.candidate), r.budget, r.seed, r.artifact) for r in kept] == [
        ("full_fine_tuning", "200", 1, KEPT)
    ]
    assert page.items[0].units == 3
    assert page.items[0].error == pytest.approx(
        closed_campaign()
        .results_of(page.items[0].candidate, closed_campaign().design.budgets[1])[0]
        .rmse
    )


def test_pages_of_runs_follow_one_another_and_together_are_the_grid() -> None:
    use_case = listing()
    pages = [use_case(ListCampaignRunsQuery(campaign=CAMPAIGN, limit=3))]
    while pages[-1].next_cursor is not None:
        pages.append(
            use_case(ListCampaignRunsQuery(campaign=CAMPAIGN, after=pages[-1].next_cursor, limit=3))
        )

    runs = [run for page in pages for run in page.items]

    assert [len(page.items) for page in pages] == [3, 3, 2]
    assert runs == [
        CampaignRun.of(r)
        for r in sorted(
            closed_campaign(KEPT).results,
            key=lambda r: (str(r.cell.candidate), r.cell.budget.text(), r.cell.seed),
        )
    ]


@pytest.mark.parametrize(
    "parts", [("x",), ("from_scratch", "many", "1"), ("from_scratch", "50", "one")]
)
def test_a_cursor_of_another_list_is_refused(parts: tuple[str, ...]) -> None:
    with pytest.raises(InvalidCursorError):
        listing()(ListCampaignRunsQuery(campaign=CAMPAIGN, after=Cursor(parts), limit=3))


def test_the_runs_of_a_campaign_nobody_stored_are_not_found() -> None:
    with pytest.raises(CampaignNotFoundError):
        listing()(ListCampaignRunsQuery(campaign=CampaignId(UUID(int=99)), limit=3))


def test_a_page_holds_at_least_one_run() -> None:
    with pytest.raises(InvalidPageError):
        ListCampaignRunsQuery(campaign=CAMPAIGN, limit=0)
