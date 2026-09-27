import logging
from collections.abc import Callable

from fastapi import FastAPI
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import CallbackOptions, Counter, Histogram, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from emblema.config.telemetry_settings import TelemetrySettings

INSTRUMENTATION = "emblema.api"


class Telemetry:
    """What the HTTP process reports about itself: traces of every request, metrics of every answer.

    OpenTelemetry throughout, so that where the reports go is configuration and never code. The
    framework's requests are traced and timed by its instrumentation; what the process adds is
    what the framework cannot see: how long the tokeniser and the model took, and how much of
    what was asked was answered. The providers are held here and handed to each instrumentation
    rather than installed as the process-wide default, so two applications in one process — a
    test composing several — report apart.
    """

    def __init__(self, settings: TelemetrySettings) -> None:
        resource = Resource.create({"service.name": settings.service_name})
        self._reader = PrometheusMetricReader()
        readers: list[MetricReader] = [self._reader]
        self._tracers = TracerProvider(resource=resource)
        if settings.otlp_endpoint is not None:
            base = settings.otlp_endpoint.rstrip("/")
            readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"))
            )
            self._tracers.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces"))
            )
        self._meters = MeterProvider(resource=resource, metric_readers=readers)
        self.tracer: Tracer = self._tracers.get_tracer(INSTRUMENTATION)
        meter = self._meters.get_meter(INSTRUMENTATION)
        self.inference_seconds: Histogram = meter.create_histogram(
            "emblema_inference_duration_seconds",
            unit="s",
            description="How long a served model took to answer or represent one request.",
        )
        self.tokenisation_seconds: Histogram = meter.create_histogram(
            "emblema_tokenisation_duration_seconds",
            unit="s",
            description="How long one window of raw readings took to tokenise.",
        )
        self.windows_answered: Counter = meter.create_counter(
            "emblema_windows_answered", description="Windows a served model answered."
        )
        self.inference_refused: Counter = meter.create_counter(
            "emblema_inference_refused",
            description="Requests refused because the networks were running all the budget allows.",
        )
        self.channels_ignored: Counter = meter.create_counter(
            "emblema_channels_ignored",
            description="Readings dropped because the model did not know their channel.",
        )

    def observe_budget(
        self, *, capacity: int, in_use: Callable[[], int], waiting: Callable[[], int]
    ) -> None:
        """Report how much of the inference budget is held and how many batches wait for it.

        Observed when scraped rather than recorded as it changes, so the report costs nothing
        to the batches themselves.
        """
        meter = self._meters.get_meter(INSTRUMENTATION)

        def reading(read: Callable[[], int]) -> Callable[[CallbackOptions], list[Observation]]:
            return lambda _: [Observation(read())]

        meter.create_observable_gauge(
            "emblema_inference_budget_capacity",
            callbacks=[reading(lambda: capacity)],
            description="Token pairs the networks may run at once.",
        )
        meter.create_observable_gauge(
            "emblema_inference_budget_in_use",
            callbacks=[reading(in_use)],
            description="Token pairs the networks are running now.",
        )
        meter.create_observable_gauge(
            "emblema_inference_batches_waiting",
            callbacks=[reading(waiting)],
            description="Batches waiting to be admitted to the networks.",
        )

    def instrument(self, app: FastAPI) -> None:
        """Trace and time every request of ``app``."""
        FastAPIInstrumentor.instrument_app(
            app, tracer_provider=self._tracers, meter_provider=self._meters
        )

    def instrument_process(self, log_level: str) -> None:  # pragma: no cover - process-wide
        """Trace the database and the object store, and stamp every log record with its trace.

        Process-wide by what each instrumentation patches, so called once, by the process that
        serves, before its composition root opens an engine; a test never calls it.
        """
        SQLAlchemyInstrumentor().instrument(
            tracer_provider=self._tracers, meter_provider=self._meters
        )
        BotocoreInstrumentor().instrument(tracer_provider=self._tracers)
        LoggingInstrumentor().instrument(
            tracer_provider=self._tracers,
            set_logging_format=True,
            log_level=logging.getLevelNamesMapping()[log_level],
        )

    @staticmethod
    def exposition() -> tuple[bytes, str]:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

    def shutdown(self) -> None:
        """Flush what is pending and give the process-wide registry back."""
        self._meters.shutdown()
        self._tracers.shutdown()
