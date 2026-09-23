from pathlib import Path
from typing import Self

from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.config.settings import Settings
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.workers.campaign_process import CampaignProcess
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

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
            workspace: Directory corpus blocks are fetched to and mapped from.
            corpora: Directory the raw corpora sit in, where a task's labels are read from.
            schedule: How long every cell learns, in how large a step — the compute budget the
                campaign's neural candidates are held to.
            lora: The low-rank updates the arm of that name adds; needed unless a provider is
                given instead.
            backbone: Weights the pretrained arms of this process's campaigns start from;
                needed unless a provider is given instead.
            device: Where a cell computes; this machine's accelerator unless given.
            store: Artifact store; the configured S3-compatible bucket unless given.
            tasks: Registry of tasks; the configured metadata database unless given.
            campaigns: Registry of campaigns; the same database unless given.
            candidates: Who competes; the four arms over ``backbone`` unless given.
            jobs: Where cells are submitted; the configured broker unless given.
            subscriptions: Where handlers are registered; a fresh registry unless given.
            events: Where use cases publish; synchronous over ``subscriptions`` unless given.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.

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
            self._arms(process, backbone, lora, schedule, device)
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
        return cls(
            settings,
            workspace=worker.workspace,
            corpora=worker.corpora,
            schedule=cls.schedule_of(worker.require_schedule()),
            lora=cls.lora_of(worker.require_lora()),
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
    def schedule_of(settings: ScheduleSettings) -> AdaptationSchedule:
        """How every cell of this process's campaigns learns, as the environment declares it.

        Raises:
            InvalidAdaptationScheduleError: If what it declares is not a schedule that stands up.
        """
        return AdaptationSchedule(
            epochs=settings.epochs,
            min_steps=settings.min_steps,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            warmup_fraction=settings.warmup_fraction,
            final_lr_fraction=settings.final_lr_fraction,
        )

    @staticmethod
    def lora_of(settings: LoraSettings) -> LoraSpec:
        """The low-rank updates the arm of that name adds, as the environment declares them.

        Raises:
            InvalidLoraSpecError: If what it declares is not a specification that stands up.
        """
        return LoraSpec(
            rank=settings.rank,
            alpha=settings.alpha,
            dropout=settings.dropout,
            targets=tuple(
                target.strip() for target in settings.targets.split(",") if target.strip()
            ),
        )

    @staticmethod
    def _arms(
        process: CampaignProcess,
        backbone: ArtifactRef | None,
        lora: LoraSpec | None,
        schedule: AdaptationSchedule,
        device: str | None,
    ) -> CandidateProvider:
        """The four ways of using the backbone this process serves.

        Raises:
            ValueError: If the process was left to build them without a backbone and updates.
        """
        if backbone is None or lora is None:
            raise ValueError(
                "without a backbone and its low-rank updates the process needs its candidates "
                "given, not built"
            )
        return BackboneCandidateProvider(
            KnownArms.over(backbone, lora),
            schedule,
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
