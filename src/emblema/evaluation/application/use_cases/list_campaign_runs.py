from dataclasses import dataclass

from emblema.evaluation.application.read_models.campaign_run import CampaignRun
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.ports.campaign_listing import CampaignListing
from emblema.shared.kernel.exceptions import InvalidCursorError, InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.paging.page import Page


@dataclass(frozen=True, kw_only=True)
class ListCampaignRunsQuery:
    """Which page of one campaign's runs to show.

    Invariants: the limit is positive.

    Attributes:
        campaign: Whose runs.
        after: Where the previous page ended; ``None`` for the first page.
        limit: How many runs the page holds at most.
    """

    campaign: CampaignId
    after: Cursor | None = None
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise InvalidPageError(f"a page holds at least one run, got a limit of {self.limit}")


class ListCampaignRuns:
    """Lists the cells of one campaign that have run, a page at a time, each page naming the next.

    A grid of four candidates over four budgets under five seeds is eighty runs, which is why a
    list of them is paged. The cursor carries the coordinates of the last run shown.
    """

    def __init__(self, listing: CampaignListing) -> None:
        self._listing = listing

    def __call__(self, query: ListCampaignRunsQuery) -> Page[CampaignRun]:
        """The page, and the cursor that opens the next one where there is one.

        Raises:
            InvalidCursorError: If the cursor is not one this issued.
            CampaignNotFoundError: If no campaign is stored under that identity.
        """
        results = self._listing.results(
            query.campaign, after=self._after(query.after), limit=query.limit + 1
        )
        shown = results[: query.limit]
        return Page(
            items=tuple(CampaignRun.of(result) for result in shown),
            next_cursor=self._cursor_of(shown[-1].cell) if len(results) > query.limit else None,
        )

    @staticmethod
    def _cursor_of(cell: CampaignCell) -> Cursor:
        return Cursor((str(cell.candidate), cell.budget.text(), str(cell.seed)))

    @staticmethod
    def _after(cursor: Cursor | None) -> CampaignCell | None:
        if cursor is None:
            return None
        candidate, budget, seed = cursor.unpack(3)
        try:
            return CampaignCell(
                candidate=CandidateRef(candidate),
                budget=LabelBudget.parse(budget),
                seed=int(seed),
            )
        except ValueError as error:
            raise InvalidCursorError(f"not a cursor of this list: {cursor.parts}") from error
