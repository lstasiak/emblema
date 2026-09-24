from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.workers.campaign_process import CampaignProcess
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.adapters.candidates.classical_candidate_provider import (
    ClassicalCandidateProvider,
)
from emblema.evaluation.adapters.minirocket.minirocket_classical_runtime import (
    MiniRocketClassicalRuntime,
)
from emblema.evaluation.adapters.routing.method_routed_classical_runtime import (
    MethodRoutedClassicalRuntime,
)
from emblema.evaluation.adapters.xgboost.xgboost_classical_runtime import XgboostClassicalRuntime
from emblema.evaluation.application.use_cases.run_classical_fit import RunClassicalFit
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


class CompositionRoot:
    """Assembles the worker that competes the baselines: the campaign process plus what it fits.

    Nothing here reaches the context that trains backbones and nothing here imports the
    machine-learning stack, which is the point rather than an accident of what it happens to
    need: a campaign made only of these runs end to end without that stack being installed, and
    the image this process ships in is the smaller for it.

    There is no backbone to hold, so assembling this costs a connection and nothing else.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use cases, each already holding its dependencies.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        workspace: Path,
        corpora: Path,
        boosting: GradientBoostingSpec | None = None,
        convolutions: RandomConvolutions | None = None,
        sources: tuple[TaskId, ...] = (),
        store: ArtifactStore | None = None,
        tasks: DownstreamTaskRepository | None = None,
        campaigns: EvaluationCampaignRepository | None = None,
        promotables: PromotableArtifactRepository | None = None,
        candidates: CandidateProvider | None = None,
        jobs: JobQueue | None = None,
        subscriptions: InMemoryEventSubscriber | None = None,
        events: EventPublisher | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        What is this root's own is ``boosting`` and ``convolutions``, how hard the baselines fit,
        and ``sources``, the tasks the layout-independent baseline may draw labels from as well;
        all are needed unless ``candidates`` is given instead of the baselines they build.
        Everything else is what ``CampaignProcess`` takes and reaches it unchanged.

        Raises:
            ValueError: If an adapter is left to the root without settings to build it from, or
                the candidates are left to it without the knobs to build them over.
        """
        process = CampaignProcess(
            settings,
            workspace=workspace,
            corpora=corpora,
            store=store,
            tasks=tasks,
            campaigns=campaigns,
            promotables=promotables,
            jobs=jobs,
            subscriptions=subscriptions,
            events=events,
            clock=clock,
            ids=ids,
        )
        self.adapters, self.services = process.assemble(
            self.baselines_over(process, boosting, convolutions, sources)
            if candidates is None
            else candidates
        )

    @classmethod
    def from_environment(cls, jobs: JobQueue | None = None) -> Self:  # pragma: no cover - env
        """The process the environment describes, submitting through ``jobs`` where one is given.

        Raises:
            ValueError: If the settings name no worker, store, database or broker, or leave out
                the classical knobs this process cannot fit without.
        """
        settings = Settings()
        worker = settings.require_worker()
        return cls(
            settings,
            workspace=worker.workspace,
            corpora=worker.corpora,
            boosting=DeclaredWorker(worker).boosting(),
            convolutions=DeclaredWorker(worker).convolutions(),
            jobs=jobs,
        )

    @classmethod
    def over(
        cls,
        *,
        store: ArtifactStore,
        tasks: DownstreamTaskRepository,
        campaigns: EvaluationCampaignRepository,
        promotables: PromotableArtifactRepository | None = None,
        candidates: CandidateProvider,
        jobs: JobQueue,
        workspace: Path,
        corpora: Path,
        subscriptions: InMemoryEventSubscriber | None = None,
        events: EventPublisher | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> Self:
        """The process over adapters of its own, nothing read from the environment."""
        return cls(
            workspace=workspace,
            corpora=corpora,
            store=store,
            tasks=tasks,
            campaigns=campaigns,
            promotables=(
                InMemoryPromotableArtifactRepository() if promotables is None else promotables
            ),
            candidates=candidates,
            jobs=jobs,
            subscriptions=subscriptions,
            events=events,
            clock=clock,
            ids=ids,
        )

    @staticmethod
    def baselines_over(
        process: CampaignProcess,
        boosting: GradientBoostingSpec | None,
        convolutions: RandomConvolutions | None,
        sources: tuple[TaskId, ...],
    ) -> CandidateProvider:
        """The baselines every campaign competes, over the parts a campaign process holds.

        Raises:
            ValueError: If the process was left to build them without the knobs to fit them by.
        """
        if boosting is None or convolutions is None:
            raise ValueError(
                "without the classical knobs the process needs its candidates given, not built"
            )
        return ClassicalCandidateProvider(
            KnownBaselines.catalogue(boosting, convolutions, sources),
            RunClassicalFit(
                process.tasks,
                process.draw_run_labels,
                process.draw_label_budget,
                MethodRoutedClassicalRuntime(
                    trees=XgboostClassicalRuntime(process.blocks, store=process.store),
                    convolutions=MiniRocketClassicalRuntime(process.blocks, store=process.store),
                ),
            ),
        )
