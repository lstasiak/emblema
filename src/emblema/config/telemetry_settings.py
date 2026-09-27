from pydantic import BaseModel, Field


class TelemetrySettings(BaseModel):
    """Where a process reports what it does, read as ``EMBLEMA_TELEMETRY__<FIELD>``.

    Metrics are always kept, and served by the process itself for a scraper; traces and metrics
    leave the process only where a collector is named, so a developer's machine reports nowhere
    by default and a deployment says where. The name has no default: every process reports under
    its own, and one borrowed from another process would merge two services in every dashboard.
    """

    service_name: str = Field(description="How the process names itself in what it reports.")
    otlp_endpoint: str | None = Field(
        default=None,
        description=(
            "Base address of the collector traces and metrics are exported to over OTLP/HTTP, "
            "e.g. http://otel-collector:4318; none by default."
        ),
    )
