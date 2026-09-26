from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from anyio import to_thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from emblema.config.api_settings import ApiSettings
from emblema.entrypoints.api.health_routes import HealthRoutes
from emblema.entrypoints.api.problem_details import ProblemDetails
from emblema.entrypoints.api.readiness import Readiness
from emblema.entrypoints.api.request_size_limit import RequestSizeLimit
from emblema.entrypoints.api.services import Services
from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.evaluation.api.campaign_routes import CampaignRoutes
from emblema.evaluation.api.evaluation_refusals import EvaluationRefusals
from emblema.serving.api.served_model_routes import ServedModelRoutes
from emblema.serving.api.serving_refusals import ServingRefusals
from emblema.shared.api.page_request import PageRequests

DESCRIPTION = (
    "Predictions and representations from whichever candidate a finished comparison measured, "
    "and the comparisons themselves: campaigns, their curves and their verdicts."
)


class EmblemaApi:
    """The HTTP application: each context's router, and what belongs to the process alone.

    A context publishes its router and the statuses of its refusals; the process adds what no
    context owns: probes, metrics, the browser's cross-origin handshake, the ceiling on a body,
    the one error shape and how many requests are answered at once. Nothing here is module
    state, so two applications in one process — a test composing several — hold their own
    services and their own telemetry.

    Attributes:
        app: The application a server runs.
    """

    def __init__(
        self,
        services: Services,
        readiness: Readiness,
        settings: ApiSettings,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._threads = settings.request_threads
        self.app = FastAPI(
            title="Emblema",
            version=version("emblema"),
            description=DESCRIPTION,
            lifespan=self._lifespan,
        )
        pages = PageRequests(
            default_size=settings.default_page_size, max_size=settings.max_page_size
        )
        ProblemDetails((*ServingRefusals.STATUSES, *EvaluationRefusals.STATUSES)).register(self.app)
        self.app.include_router(HealthRoutes(readiness, telemetry).router)
        self.app.include_router(
            ServedModelRoutes(
                predict=services.predict_windows,
                embed=services.embed_windows,
                view=services.view_served_model,
                listed=services.list_served_models,
                pages=pages,
            ).router
        )
        self.app.include_router(
            CampaignRoutes(
                listed=services.list_campaigns,
                view=services.view_campaign,
                runs=services.list_campaign_runs,
                pages=pages,
            ).router
        )
        self.app.add_middleware(RequestSizeLimit, max_bytes=settings.max_request_bytes)
        origins = settings.origins()
        if origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=list(origins),
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        if telemetry is not None:
            telemetry.instrument(self.app)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        # Every route is synchronous and runs on the server's thread pool, so its size is how
        # many requests the process answers at once; the database pool is sized to match.
        to_thread.current_default_thread_limiter().total_tokens = self._threads
        try:
            yield
        finally:
            if self._telemetry is not None:
                self._telemetry.shutdown()
