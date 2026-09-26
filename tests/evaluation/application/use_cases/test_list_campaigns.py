from datetime import timedelta
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.campaign_listing import InMemoryCampaignListing
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.list_campaigns import (
    ListCampaigns,
    ListCampaignsQuery,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import OPENED_AT, campaign

IDS = tuple(CampaignId(UUID(int=number)) for number in (1, 2, 3))


def listing() -> ListCampaigns:
    repository = InMemoryEvaluationCampaignRepository()
    for hours, campaign_id in enumerate(IDS):
        repository.save(
            campaign(
                campaign_id=campaign_id,
                opened_at=UtcDateTime(OPENED_AT.value + timedelta(hours=hours)),
            ),
            seen=0,
        )
    return ListCampaigns(InMemoryCampaignListing(repository))


def test_campaigns_are_listed_newest_first_and_a_short_list_ends_the_page() -> None:
    page = listing()(ListCampaignsQuery(limit=10))

    assert [c.campaign_id for c in page.items] == [IDS[2], IDS[1], IDS[0]]
    assert page.next_cursor is None
    assert page.items[0].cells_planned == 8
    assert page.items[0].cells_recorded == 0


def test_a_page_names_the_next_one_and_the_pages_together_are_the_list() -> None:
    first = listing()(ListCampaignsQuery(limit=2))
    assert first.next_cursor is not None
    second = listing()(ListCampaignsQuery(after=first.next_cursor, limit=2))

    assert [c.campaign_id for c in first.items] == [IDS[2], IDS[1]]
    assert [c.campaign_id for c in second.items] == [IDS[0]]
    assert second.next_cursor is None


@pytest.mark.parametrize(
    "parts",
    [
        ("a", "b", "c"),
        (str(IDS[0]),),
        ("not-a-time", str(IDS[0])),
        ("2026-01-01T00:00:00", str(IDS[0])),
        (OPENED_AT.value.isoformat(), "not-a-uuid"),
    ],
)
def test_a_cursor_this_list_did_not_issue_is_refused(parts: tuple[str, ...]) -> None:
    with pytest.raises(InvalidCursorError):
        listing()(ListCampaignsQuery(after=Cursor(parts), limit=2))


def test_a_page_continues_after_its_position_whether_or_not_that_campaign_is_still_stored() -> None:
    position = UtcDateTime(OPENED_AT.value + timedelta(hours=2)).value.isoformat()
    after_nobody = Cursor((position, str(UUID(int=99))))

    page = listing()(ListCampaignsQuery(after=after_nobody, limit=5))

    assert [c.campaign_id for c in page.items] == [IDS[1], IDS[0]]


def test_a_page_holds_at_least_one_campaign() -> None:
    with pytest.raises(InvalidPageError):
        ListCampaignsQuery(limit=0)
