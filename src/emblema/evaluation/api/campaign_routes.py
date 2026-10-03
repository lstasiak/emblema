from collections.abc import Callable
from http import HTTPStatus
from typing import Any, ClassVar

from fastapi import APIRouter

from emblema.evaluation.api.schemas.campaign_detail import CampaignDetail
from emblema.evaluation.api.schemas.campaign_resource import CampaignResource
from emblema.evaluation.api.schemas.campaign_run_resource import CampaignRunResource
from emblema.evaluation.application.read_models.campaign_run import CampaignRun
from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary
from emblema.evaluation.application.read_models.campaign_view import CampaignView
from emblema.evaluation.application.use_cases.list_campaign_runs import ListCampaignRunsQuery
from emblema.evaluation.application.use_cases.list_campaigns import ListCampaignsQuery
from emblema.evaluation.application.use_cases.view_campaign import ViewCampaignQuery
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.shared.api.cursor_page import CursorPage
from emblema.shared.api.page_request import CursorParameter, LimitParameter, PageRequests
from emblema.shared.api.problem import Problem
from emblema.shared.kernel.paging.page import Page


class CampaignRoutes:
    """What the API shows of evaluation campaigns: a list, one in full, and its runs.

    Read only: a campaign is declared and advanced from the command line, and what the API adds
    is a way to read the verdict, the curves and the grid without the repository at hand.

    Attributes:
        router: The routes, under ``/campaigns``.
    """

    _REFUSALS: ClassVar[dict[int | str, dict[str, Any]]] = Problem.responses(
        HTTPStatus.NOT_FOUND, HTTPStatus.UNPROCESSABLE_ENTITY
    )

    def __init__(
        self,
        *,
        listed: Callable[[ListCampaignsQuery], Page[CampaignSummary]],
        view: Callable[[ViewCampaignQuery], CampaignView],
        runs: Callable[[ListCampaignRunsQuery], Page[CampaignRun]],
        pages: PageRequests,
    ) -> None:
        self._listed = listed
        self._view = view
        self._runs = runs
        self._pages = pages
        self.router = APIRouter(prefix="/campaigns", tags=["campaigns"])
        self.router.get(
            "",
            operation_id="list_campaigns",
            summary="Campaigns, newest first",
            response_model=CursorPage[CampaignResource],
            responses=Problem.responses(HTTPStatus.UNPROCESSABLE_ENTITY),
        )(self.list_campaigns)
        self.router.get(
            "/{campaign_id}",
            operation_id="view_campaign",
            summary="One campaign: its design, every candidate's curve, and the verdict",
            response_model=CampaignDetail,
            responses=self._REFUSALS,
        )(self.view_campaign)
        self.router.get(
            "/{campaign_id}/runs",
            operation_id="list_campaign_runs",
            summary="The cells of a campaign that have run",
            response_model=CursorPage[CampaignRunResource],
            responses=self._REFUSALS,
        )(self.list_runs)

    def list_campaigns(
        self, cursor: CursorParameter = None, limit: LimitParameter = None
    ) -> CursorPage[CampaignResource]:
        page = self._pages.read(cursor, limit)
        listed = self._listed(ListCampaignsQuery(after=page.after, limit=page.limit))
        return CursorPage[CampaignResource].of(listed, CampaignResource.of)

    def view_campaign(self, campaign_id: str) -> CampaignDetail:
        view = self._view(ViewCampaignQuery(campaign=CampaignId.parse(campaign_id)))
        return CampaignDetail.of(view)

    def list_runs(
        self, campaign_id: str, cursor: CursorParameter = None, limit: LimitParameter = None
    ) -> CursorPage[CampaignRunResource]:
        page = self._pages.read(cursor, limit)
        runs = self._runs(
            ListCampaignRunsQuery(
                campaign=CampaignId.parse(campaign_id), after=page.after, limit=page.limit
            )
        )
        return CursorPage[CampaignRunResource].of(runs, CampaignRunResource.of)
