from collections.abc import Callable
from http import HTTPStatus

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from emblema.entrypoints.api.readiness import Readiness
from emblema.entrypoints.api.schemas.health_resource import HealthResource
from emblema.entrypoints.api.schemas.readiness_resource import ReadinessResource
from emblema.entrypoints.api.telemetry.telemetry import Telemetry


class HealthRoutes:
    """The probes an orchestrator asks, and the metrics a scraper reads, as one router.

    The metrics are served here rather than on a port of their own, because there is one
    process and one port to publish, and only where telemetry was composed in: a process
    without it has nothing to expose.

    Attributes:
        router: The routes, at the root of the path.
    """

    def __init__(self, readiness: Readiness, telemetry: Telemetry | None) -> None:
        self._readiness = readiness
        self.router = APIRouter(tags=["operations"])
        self.router.get(
            "/health",
            operation_id="health",
            summary="Liveness: the process is up",
            response_model=HealthResource,
        )(self.health)
        self.router.get(
            "/ready",
            operation_id="ready",
            summary="Readiness: every dependency answers",
            response_model=ReadinessResource,
            responses={HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ReadinessResource}},
        )(self.ready)
        if telemetry is not None:
            self.router.get("/metrics", operation_id="metrics", include_in_schema=False)(
                self._metrics_of(telemetry)
            )

    def health(self) -> HealthResource:
        return HealthResource(status="ok")

    def ready(self) -> JSONResponse:
        report = self._readiness.report()
        status = HTTPStatus.OK if report.ready else HTTPStatus.SERVICE_UNAVAILABLE
        return JSONResponse(
            status_code=status.value, content=ReadinessResource.of(report).model_dump()
        )

    @staticmethod
    def _metrics_of(telemetry: Telemetry) -> Callable[[], Response]:
        def metrics() -> Response:
            body, media_type = telemetry.exposition()
            return Response(content=body, media_type=media_type)

        return metrics
