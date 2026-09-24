from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.workers.campaign_process import CampaignProcess
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.evaluation.adapters.candidates.backbone_candidate_provider import (
    BackboneCandidateProvider,
)
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import TorchAdaptationRuntime
from emblema.evaluation.application.use_cases.run_adaptation import RunAdaptation
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


class CompositionRoot:
    """Assembles the worker that adapts a backbone: the campaign process plus the arms it serves.

    The process is told which backbone it serves, holds it, and refuses a cell of a campaign
    that was run over other weights. Holding it is the reason this root exists apart from the
    one that fits trees: reading a backbone out of the store for each of a grid's cells would
    download it eighty times to save keeping one copy, and everything else the two processes
    need is the same and is assembled once.

    This is the image with the machine-learning stack in it. Whether a cell computes on an
    accelerator is the device's business and not this process's: the target platform is a
    container that has none, and the same entry point on a host that does is a way of reaching
    one rather than a second architecture.

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
        schedule: AdaptationSchedule,
        lora: LoraSpec | None = None,
        backbone: ArtifactRef | None = None,
        device: str | None = None,
        store: ArtifactStore | None = None,
        tasks: DownstreamTaskRepository | None = None,
        campaigns: EvaluationCampaignRepository | None = None,
        candidates: CandidateProvider | None = None,
        jobs: JobQueue | None = None,
        subscriptions: InMemoryEventSubscriber | None = None,
        events: EventPublisher | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        What is this root's own is the ``schedule`` every cell learns under — the compute budget
        the campaign's neural candidates are held to — the ``backbone`` the pretrained arms
        start from, the ``lora`` updates the arm of that name adds, and the ``device`` a cell
        computes on, this machine's accelerator unless given. Backbone and updates are needed
        unless ``candidates`` is given instead of the four arms they build. Everything else is
        what ``CampaignProcess`` takes and reaches it unchanged.

        Raises:
            ValueError: If an adapter is left to the root without settings to build it from, or
                the candidates are left to it without a backbone and updates to build them over.
        """
        process = CampaignProcess(
            settings,
            workspace=workspace,
            corpora=corpora,
            store=store,
            tasks=tasks,
            campaigns=campaigns,
            jobs=jobs,
            subscriptions=subscriptions,
            events=events,
            clock=clock,
            ids=ids,
        )
        self.adapters, self.services = process.assemble(
            self.arms_over(process, backbone, lora, schedule, device)
            if candidates is None
            else candidates
        )

    @classmethod
    def from_environment(cls, jobs: JobQueue | None = None) -> Self:  # pragma: no cover - env
        """The process the environment describes, submitting through ``jobs`` where one is given.

        Raises:
            ValueError: If the settings name no worker, store, database or broker, or leave out
                the backbone, schedule or low-rank updates this process cannot run without.
        """
        settings = Settings()
        worker = settings.require_worker()
        declared = DeclaredWorker(worker)
        return cls(
            settings,
            workspace=worker.workspace,
            corpora=worker.corpora,
            schedule=declared.schedule(),
            lora=declared.lora(),
            backbone=worker.require_backbone_ref(),
            device=worker.device,
            jobs=jobs,
        )

    @classmethod
    def over(
        cls,
        *,
        store: ArtifactStore,
        tasks: DownstreamTaskRepository,
        campaigns: EvaluationCampaignRepository,
        candidates: CandidateProvider,
        jobs: JobQueue,
        workspace: Path,
        corpora: Path,
        schedule: AdaptationSchedule,
        subscriptions: InMemoryEventSubscriber | None = None,
        events: EventPublisher | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> Self:
        """The process over adapters of its own, nothing read from the environment.

        What the general constructor refuses at runtime when settings are missing, this refuses
        at the type: the five adapters it would otherwise build are required here.
        """
        return cls(
            workspace=workspace,
            corpora=corpora,
            schedule=schedule,
            store=store,
            tasks=tasks,
            campaigns=campaigns,
            candidates=candidates,
            jobs=jobs,
            subscriptions=subscriptions,
            events=events,
            clock=clock,
            ids=ids,
        )

    @staticmethod
    def arms_over(
        process: CampaignProcess,
        backbone: ArtifactRef | None,
        lora: LoraSpec | None,
        schedule: AdaptationSchedule,
        device: str | None,
    ) -> CandidateProvider:
        """The four ways of using a backbone, over the parts a campaign process holds.

        Raises:
            ValueError: If the process was left to build them without a backbone and updates.
        """
        if backbone is None or lora is None:
            raise ValueError(
                "without a backbone and its low-rank updates the process needs its candidates "
                "given, not built"
            )
        return BackboneCandidateProvider(
            KnownArms.catalogue(backbone, lora, schedule),
            RunAdaptation(
                process.draw_run_labels,
                TorchAdaptationRuntime(
                    RestoredBackbones(process.store, backbone),
                    process.blocks,
                    device=device or available_device(),
                    store=process.store,
                ),
            ),
        )
