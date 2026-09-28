from typing import Protocol

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.campaign.cell_result import CellResult


class CampaignListing(Protocol):
    """Lists campaigns, and the cells of one, a page at a time.

    The read side of the registry, apart from the repository that keeps one campaign at a time:
    a list is asked of the whole table, and a grid of eighty cells is read in pages. A page
    continues after the position of the last row of the one before, so a cell recorded while a
    client reads shifts nothing the client has already seen.
    """

    def campaigns(
        self, *, after: CampaignPosition | None, limit: int
    ) -> tuple[CampaignOverview, ...]:
        """At most ``limit`` campaigns following ``after``, or the first ones where it is ``None``.

        Ordered by when they were designed, latest first, and by identity where two were
        designed at once; each without its results, which a list does not show.
        """
        ...

    def get_overview(self, campaign: CampaignId) -> CampaignOverview:
        """One campaign without its results: its design, and how many cells of it have run.

        Raises:
            CampaignNotFoundError: If no campaign is stored under that identity.
        """
        ...

    def results(
        self, campaign: CampaignId, *, after: CampaignCell | None, limit: int
    ) -> tuple[CellResult, ...]:
        """At most ``limit`` recorded cells of ``campaign`` following ``after``.

        Ordered by candidate, budget and seed as their text sorts, which is one order for every
        adapter and not the grid's own.

        Raises:
            CampaignNotFoundError: If no campaign is stored under that identity.
        """
        ...
