from dataclasses import dataclass
from datetime import datetime

from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.ports.campaign_listing import CampaignListing
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.paging.page import Page
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class ListCampaignsQuery:
    """Which page of campaigns to show.

    Invariants: the limit is positive.

    Attributes:
        after: Where the previous page ended; ``None`` for the first page.
        limit: How many campaigns the page holds at most.
    """

    after: Cursor | None = None
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise InvalidPageError(
                f"a page holds at least one campaign, got a limit of {self.limit}"
            )


class ListCampaigns:
    """Lists campaigns newest first, a page at a time, each page naming the next.

    The cursor carries the position of the last campaign shown — when it was designed and its
    identity — so the next page is read after it without looking the campaign up; one more
    campaign than the page holds is read, so that the page knows whether there is a next one
    without a count.
    """

    def __init__(self, listing: CampaignListing) -> None:
        self._listing = listing

    def __call__(self, query: ListCampaignsQuery) -> Page[CampaignSummary]:
        """The page, and the cursor that opens the next one where there is one.

        Raises:
            InvalidCursorError: If the cursor is not one this issued.
        """
        after = None if query.after is None else self._position(query.after)
        campaigns = self._listing.campaigns(after=after, limit=query.limit + 1)
        shown = campaigns[: query.limit]
        return Page(
            items=tuple(CampaignSummary.of(campaign) for campaign in shown),
            next_cursor=(
                self._cursor(shown[-1].position) if len(campaigns) > query.limit else None
            ),
        )

    @staticmethod
    def _cursor(position: CampaignPosition) -> Cursor:
        return Cursor((position.opened_at.value.isoformat(), str(position.campaign_id)))

    @staticmethod
    def _position(cursor: Cursor) -> CampaignPosition:
        opened_at, campaign_id = cursor.unpack(2)
        try:
            return CampaignPosition(
                opened_at=UtcDateTime(datetime.fromisoformat(opened_at)),
                campaign_id=CampaignId.parse(campaign_id),
            )
        except ValueError as error:
            raise InvalidCursorError(f"not a cursor of this list: {cursor.parts}") from error
