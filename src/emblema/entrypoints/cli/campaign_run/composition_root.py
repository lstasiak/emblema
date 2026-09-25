from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign_run.adapters import Adapters
from emblema.entrypoints.cli.campaign_run.services import Services
from emblema.entrypoints.configured import configured_store, settings_for
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.evaluation.adapters.handoff.artifact_store_campaign_handoff import (
    ArtifactStoreCampaignHandoff,
)
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.fulfil_campaign_order import FulfilCampaignOrder
from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class CompositionRoot:
    """The process that runs an order of campaign cells where no registry and no queue reach.

    Which candidates it carries is decided by the order, not by a flag: an order is one pool's
    cells, and the root that pool's worker is assembled from is assembled here too, over a
    registry of this run instead of the database and a queue that runs nothing. So a cell run
    through an order is answered by the very candidates a worker would answer it with.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use case it runs.
    """

    def __init__(
        self,
        *,
        handoff: CampaignHandoff,
        tasks: DownstreamTaskRepository,
        candidates: CandidateProvider,
    ) -> None:
        self.adapters = Adapters(handoff=handoff, tasks=tasks, candidates=candidates)
        self.services = Services(
            fulfil_campaign_order=FulfilCampaignOrder(handoff, tasks, candidates)
        )

    @classmethod
    def from_environment(
        cls, order: ArtifactRef, settings: Settings | None = None
    ) -> Self:  # pragma: no cover - environment
        """The process that runs ``order``, over the store and the worker the environment names.

        The order is read once here to learn which pool it is for; the use case reads it again,
        which costs a small document and keeps the run from trusting this process's reading.

        Raises:
            ValueError: If the settings name no store or worker, or leave out anything the
                order's pool cannot run without.
        """
        read = Settings() if settings is None else settings
        store = configured_store(settings_for(read, "store"))
        handoff = ArtifactStoreCampaignHandoff(store)
        tasks = InMemoryDownstreamTaskRepository()
        needs_the_stack = handoff.read_order(order).needs_the_ml_stack
        return cls(
            handoff=handoff,
            tasks=tasks,
            candidates=cls._candidates(read, store, tasks, needs_the_stack=needs_the_stack),
        )

    @staticmethod
    def _candidates(
        settings: Settings,
        store: ArtifactStore,
        tasks: DownstreamTaskRepository,
        *,
        needs_the_stack: bool,
    ) -> CandidateProvider:  # pragma: no cover - environment
        """The candidates of the pool the order is for, as that pool's worker assembles them.

        Each root is imported only when its pool is the one asked for: on some platforms the
        stack of one cannot be loaded into a process that holds the other.
        """
        worker = settings.require_worker()
        declared = DeclaredWorker(worker)
        campaigns = InMemoryEvaluationCampaignRepository()
        jobs = ImmediateJobQueue({})
        if needs_the_stack:
            from emblema.entrypoints.workers.ml.composition_root import (
                CompositionRoot as MlRoot,
            )

            return MlRoot(
                settings,
                workspace=worker.workspace,
                corpora=worker.corpora,
                schedule=declared.schedule(),
                lora=declared.lora(),
                backbone=worker.require_backbone_ref(),
                patch=declared.patch(),
                device=worker.device,
                store=store,
                tasks=tasks,
                campaigns=campaigns,
                jobs=jobs,
            ).adapters.candidates
        from emblema.entrypoints.workers.general.composition_root import (
            CompositionRoot as GeneralRoot,
        )

        return GeneralRoot(
            settings,
            workspace=worker.workspace,
            corpora=worker.corpora,
            boosting=declared.boosting(),
            convolutions=declared.convolutions(),
            store=store,
            tasks=tasks,
            campaigns=campaigns,
            jobs=jobs,
        ).adapters.candidates
