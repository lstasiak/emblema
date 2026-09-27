from dataclasses import replace

import pytest

from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.exceptions import InvalidCampaignOverviewError
from tests.evaluation.support import CAMPAIGN, CLOSED_AT, OPENED_AT, campaign, closed_reading


def test_an_overview_counts_the_cells_a_campaign_ran_and_says_where_it_stands() -> None:
    closed = closed_reading()

    overview = CampaignOverview.of(closed.campaign)

    assert overview.cells_recorded == len(closed.results) == len(closed.campaign.design.cells())
    assert overview.is_finished
    assert overview.position == CampaignPosition(opened_at=OPENED_AT, campaign_id=CAMPAIGN)


def test_an_overview_counts_no_more_cells_than_the_grid_holds() -> None:
    overview = CampaignOverview.of(campaign())

    with pytest.raises(InvalidCampaignOverviewError, match="grid"):
        replace(overview, cells_recorded=len(overview.design.cells()) + 1)
    with pytest.raises(InvalidCampaignOverviewError, match="grid"):
        replace(overview, cells_recorded=-1)


def test_a_finished_campaign_ran_every_cell() -> None:
    overview = CampaignOverview.of(campaign())

    with pytest.raises(InvalidCampaignOverviewError, match="pending"):
        replace(overview, completed_at=CLOSED_AT)
