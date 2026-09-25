from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.workers.campaign_process import CampaignProcess
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.known_patch_models import KnownPatchModels
from emblema.evaluation.adapters.candidates.backbone_candidate_provider import (
    BackboneCandidateProvider,
)
from emblema.evaluation.adapters.candidates.patch_candidate_provider import (
    PatchCandidateProvider,
)
from emblema.evaluation.adapters.candidates.routed_candidate_provider import (
    RoutedCandidateProvider,
)
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import TorchAdaptationRuntime
from emblema.evaluation.adapters.torch.torch_patch_runtime import TorchPatchRuntime
from emblema.evaluation.application.use_cases.run_adaptation import RunAdaptation
from emblema.evaluation.application.use_cases.run_patch_training import RunPatchTraining
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.pretraining.adapters.training.devices import available_device
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


class CompositionRoot:
    """Assembles the worker that trains networks: the campaign process plus what it competes.

    Two kinds of network come from here: the arms made out of a backbone, and a patch model
    trained from nothing on a grid. They share the compute budget and the stack, which is why one
    image runs both, and the process routes each name to whichever supplies it.

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
        patch: PatchModelSpec | None = None,
        device: str | None = None,
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

        What is this root's own is the ``schedule`` every cell learns under — the compute budget
        the campaign's neural candidates are held to — the ``backbone`` the pretrained arms
        start from, the ``lora`` updates the arm of that name adds, the ``patch`` model's shape,
        and the ``device`` a cell computes on, this machine's accelerator unless given. Backbone,
        updates and shape are needed unless ``candidates`` is given instead of the networks they
        build. Everything else is what ``CampaignProcess`` takes and reaches it unchanged.

        Raises:
            ValueError: If an adapter is left to the root without settings to build it from, or
                the candidates are left to it without a backbone, updates and a shape to build
                them over.
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
            self.networks_over(process, backbone, lora, patch, schedule, device)
            if candidates is None
            else candidates
        )

    @classmethod
    def from_environment(cls, jobs: JobQueue | None = None) -> Self:  # pragma: no cover - env
        """The process the environment describes, submitting through ``jobs`` where one is given.

        Raises:
            ValueError: If the settings name no worker, store, database or broker, or leave out
                the backbone, schedule, low-rank updates or patch model this process cannot run
                without.
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
            patch=declared.patch(),
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
        promotables: PromotableArtifactRepository | None = None,
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
    def networks_over(
        process: CampaignProcess,
        backbone: ArtifactRef | None,
        lora: LoraSpec | None,
        patch: PatchModelSpec | None,
        schedule: AdaptationSchedule,
        device: str | None,
    ) -> CandidateProvider:
        """The four ways of using a backbone and the patch model, over what the process holds.

        Both learn under one ``schedule`` on one ``device``, which is what holds them to the same
        compute budget.

        Raises:
            ValueError: If the process was left to build them without a backbone, updates and a
                shape for the patch model.
        """
        if backbone is None or lora is None or patch is None:
            raise ValueError(
                "without a backbone, its low-rank updates and a patch model's shape the process "
                "needs its candidates given, not built"
            )
        on = device or available_device()
        arms = BackboneCandidateProvider(
            KnownArms.catalogue(backbone, lora, schedule),
            RunAdaptation(
                process.draw_run_labels,
                TorchAdaptationRuntime(
                    RestoredBackbones(process.store, backbone),
                    process.blocks,
                    device=on,
                    store=process.store,
                ),
            ),
        )
        patched = PatchCandidateProvider(
            KnownPatchModels.catalogue(patch, schedule),
            RunPatchTraining(
                process.draw_run_labels,
                TorchPatchRuntime(process.blocks, device=on, store=process.store),
            ),
        )
        return RoutedCandidateProvider(
            {
                **dict.fromkeys(KnownArms.refs(), arms),
                **dict.fromkeys(KnownPatchModels.refs(), patched),
            }
        )
