from collections.abc import Callable

from sqlalchemy import Engine, text

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.catalog.contracts.window_tokeniser import WindowTokeniser
from emblema.config.api_settings import ApiSettings
from emblema.config.settings import Settings
from emblema.entrypoints.api.adapters import Adapters
from emblema.entrypoints.api.readiness import Readiness
from emblema.entrypoints.api.services import Services
from emblema.entrypoints.api.telemetry.counted_answers import CountedAnswers
from emblema.entrypoints.api.telemetry.instrumented_inference_runtime import (
    InstrumentedInferenceRuntime,
)
from emblema.entrypoints.api.telemetry.instrumented_window_tokeniser import (
    InstrumentedWindowTokeniser,
)
from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.entrypoints.configured import configured_engine, configured_store, settings_for
from emblema.evaluation.adapters.artifacts.kept_classical_inference import KeptClassicalInference
from emblema.evaluation.adapters.in_memory.verdict_memo import InMemoryVerdictMemo
from emblema.evaluation.adapters.persistence.campaign_listing import SqlAlchemyCampaignListing
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.list_campaign_runs import ListCampaignRuns
from emblema.evaluation.application.use_cases.list_campaigns import ListCampaigns
from emblema.evaluation.application.use_cases.view_campaign import ViewCampaign
from emblema.evaluation.ports.campaign_listing import CampaignListing
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.evaluation.ports.verdict_memo import VerdictMemo
from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference
from emblema.serving.adapters.persistence.served_model_listing import SqlAlchemyServedModelListing
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.serving.adapters.routing.format_routed_inference_runtime import (
    FormatRoutedInferenceRuntime,
)
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.application.use_cases.embed_windows import EmbedWindows
from emblema.serving.application.use_cases.list_served_models import ListServedModels
from emblema.serving.application.use_cases.predict_windows import PredictWindows
from emblema.serving.application.use_cases.view_served_model import ViewServedModel
from emblema.serving.domain.inference_limits import InferenceLimits
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.serving.ports.served_model_listing import ServedModelListing
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore

# What the readiness probe asks the store for: a reference to nothing, so that a reachable
# bucket answers "no" and an unreachable one raises, and nothing is written either way.
_PROBE = ArtifactRef(key="readiness/probe", checksum=Checksum.of_bytes(b""))


class CompositionRoot:
    """Assembles the HTTP process: what it answers with, and what it runs on.

    Two contexts answer here. Serving answers predictions and representations and shows what it
    serves; Evaluation shows its campaigns. Both read one database and one store, so both are
    composed on one engine and one store, and neither feeds the other: what is promotable is
    decided where campaigns close, and this process closes none.

    Nothing of the training stack is composed in: a network is run through its graph, a
    classical candidate through the service the Evaluation context publishes for it, and the
    tokenisation of a request through the service the Catalog publishes. All lifetimes are
    process-scoped, the loaded candidates and the verdicts read included, which is the point of
    a service that answers many requests with the same few models.

    Attributes:
        adapters: The port implementations, and the published services, the process runs on.
        services: The use cases, each already holding its dependencies.
        readiness: What the readiness probe asks: the database and the store unless given.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        telemetry: Telemetry | None = None,
        limits: InferenceLimits | None = None,
        store: ArtifactStore | None = None,
        served: ServedModelRepository | None = None,
        served_listing: ServedModelListing | None = None,
        campaigns: EvaluationCampaignRepository | None = None,
        campaign_listing: CampaignListing | None = None,
        verdicts: VerdictMemo | None = None,
        runtime: InferenceRuntime | None = None,
        tokeniser: WindowTokeniser | None = None,
        checks: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        """Assemble the process.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
            telemetry: What the process reports itself through; nothing is reported unless given.
            limits: How much one request may ask; read from the settings unless given.
            store: Artifact store; the configured S3-compatible bucket unless given.
            served: Registry of served models; the configured database unless given.
            served_listing: Pages of served models; the same database unless given.
            campaigns: Registry of campaigns; the same database unless given.
            campaign_listing: Pages of campaigns and their runs; the same database unless given.
            verdicts: Memo of verdicts read; one of the configured capacity unless given.
            runtime: What runs a served artifact; routed by the form kept, over the store,
                with the graph runtime and the classical service composed in, unless given.
            tokeniser: The Catalog's service for one window; its adapter unless given.
            checks: What the readiness probe asks; the database and the store unless given.

        Raises:
            ValueError: If an adapter is left to the root without settings to build it from.
        """
        chosen_store = configured_store(settings_for(settings, "store")) if store is None else store
        engine = self._engine(settings, served, served_listing, campaigns, campaign_listing)
        chosen_runtime = self._runtime(settings, chosen_store) if runtime is None else runtime
        chosen_tokeniser = PublishedWindowTokeniser() if tokeniser is None else tokeniser
        if telemetry is not None:
            chosen_runtime = InstrumentedInferenceRuntime(chosen_runtime, telemetry)
            chosen_tokeniser = InstrumentedWindowTokeniser(chosen_tokeniser, telemetry)
        self.adapters = Adapters(
            store=chosen_store,
            served=self._built(served, engine, SqlAlchemyServedModelRepository),
            served_listing=self._built(served_listing, engine, SqlAlchemyServedModelListing),
            campaigns=self._built(campaigns, engine, SqlAlchemyEvaluationCampaignRepository),
            campaign_listing=self._built(campaign_listing, engine, SqlAlchemyCampaignListing),
            verdicts=(
                InMemoryVerdictMemo(capacity=self._api(settings).verdict_memo_capacity)
                if verdicts is None
                else verdicts
            ),
            runtime=chosen_runtime,
            tokeniser=chosen_tokeniser,
        )
        chosen_limits = self._limits(self._api(settings)) if limits is None else limits
        admission = WindowAdmission(self.adapters.runtime, self.adapters.tokeniser, chosen_limits)
        predict = PredictWindows(self.adapters.served, self.adapters.runtime, admission)
        embed = EmbedWindows(self.adapters.served, self.adapters.runtime, admission)
        self.services = Services(
            predict_windows=(
                predict
                if telemetry is None
                else CountedAnswers(predict, telemetry, kind="prediction")
            ),
            embed_windows=(
                embed if telemetry is None else CountedAnswers(embed, telemetry, kind="embedding")
            ),
            view_served_model=ViewServedModel(self.adapters.served, self.adapters.runtime),
            list_served_models=ListServedModels(self.adapters.served_listing),
            view_campaign=ViewCampaign(self.adapters.campaigns, self.adapters.verdicts),
            list_campaigns=ListCampaigns(self.adapters.campaign_listing),
            list_campaign_runs=ListCampaignRuns(self.adapters.campaign_listing),
        )
        self.readiness = Readiness(self._checks(engine, chosen_store) if checks is None else checks)

    @staticmethod
    def _api(settings: Settings | None) -> ApiSettings:
        return settings_for(settings, "the service's limits").require_api()

    @classmethod
    def _runtime(cls, settings: Settings | None, store: ArtifactStore) -> InferenceRuntime:
        api = cls._api(settings)
        return FormatRoutedInferenceRuntime(
            store,
            graphs=OnnxGraphInference(
                batch_size=api.batch_size, providers=api.providers(), threads=api.onnx_threads
            ),
            classical=KeptClassicalInference(),
        )

    @staticmethod
    def _limits(api: ApiSettings) -> InferenceLimits:
        return InferenceLimits(
            max_windows_per_request=api.max_windows_per_request,
            max_tokens_per_window=api.max_tokens_per_window,
        )

    @classmethod
    def _engine(cls, settings: Settings | None, *given: object) -> Engine | None:
        """One engine wherever a registry was left to be built, none where every one was given.

        The pool holds a connection for every request the process answers at once, so no
        request waits on another's.

        Raises:
            ValueError: If one was left to be built without settings to build it from.
        """
        if all(adapter is not None for adapter in given):
            return None
        return configured_engine(
            settings_for(settings, "the registries"),
            pool_size=cls._api(settings).request_threads,
        )

    @staticmethod
    def _built[T](given: T | None, engine: Engine | None, build: Callable[[Engine], T]) -> T:
        if given is not None:
            return given
        # The engine is built whenever any registry is left to the root, so this cannot be
        # reached without one; the check keeps the type honest rather than guarding a path.
        if engine is None:  # pragma: no cover
            raise ValueError("a registry was left to the root without an engine to build it on")
        return build(engine)

    @staticmethod
    def _checks(engine: Engine | None, store: ArtifactStore) -> dict[str, Callable[[], None]]:
        """What readiness asks of the real adapters: a round trip to each."""

        def artifact_store() -> None:
            store.exists(_PROBE)

        checks: dict[str, Callable[[], None]] = {"artifact_store": artifact_store}
        if engine is not None:

            def database() -> None:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))

            checks["database"] = database
        return checks
