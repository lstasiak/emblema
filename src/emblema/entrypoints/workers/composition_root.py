from pathlib import Path
from typing import Self

from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_engine, configured_store, settings_for
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.entrypoints.workers.adapters import Adapters
from emblema.entrypoints.workers.known_arms import KnownArms
from emblema.entrypoints.workers.services import Services
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.candidates.backbone_candidate_provider import (
    BackboneCandidateProvider,
)
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import TorchAdaptationRuntime
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_adaptation import RunAdaptation
from emblema.evaluation.application.use_cases.run_campaign_cell import RunCampaignCell
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


class CompositionRoot:
    """Assembles the campaign worker: every adapter chosen once, every use case built on them.

    The third process this system has, and the first that outlives a single command. Its
    lifetimes are all process-scoped, which is a decision rather than a simplification: the one
    thing worth scoping per task would be the backbone, and reading a backbone out of the store
    for each of a grid's cells would download it eighty times to save holding one copy. So the
    process is told which backbone it serves, holds it, and refuses a cell of a campaign that
    was run over other weights.

    Handlers of events are registered through the subscriber, and use cases publish through the
    publisher, so what listens is decided here and not by whoever publishes. The publisher is
    the synchronous one: nothing yet has to outlive the transaction that publishes it, and an
    outbox is the answer on the day something does.

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
        chosen_store = configured_store(settings_for(settings, "store")) if store is None else store
        chosen_subscriptions = InMemoryEventSubscriber() if subscriptions is None else subscriptions
        chosen_events = InMemoryEventPublisher(chosen_subscriptions) if events is None else events
        chosen_clock = SystemClock() if clock is None else clock
        chosen_ids = Uuid4IdGenerator() if ids is None else ids
        chosen_tasks, chosen_campaigns = self._registries(settings, tasks, campaigns)
        # Built before the provider and handed to it, so that the publisher this operation
        # records through is the process's own and can be seen to be.
        open_test_split = OpenTestSplit(chosen_tasks, chosen_ids, chosen_clock, chosen_events)
        self.adapters = Adapters(
            store=chosen_store,
            tasks=chosen_tasks,
            campaigns=chosen_campaigns,
            candidates=(
                self._candidates(
                    chosen_store,
                    chosen_tasks,
                    workspace,
                    corpora,
                    backbone,
                    lora,
                    schedule,
                    device,
                    open_test_split,
                )
                if candidates is None
                else candidates
            ),
            jobs=self._jobs(settings) if jobs is None else jobs,
            events=chosen_events,
            subscriptions=chosen_subscriptions,
            clock=chosen_clock,
            ids=chosen_ids,
        )
        self.services = self._services(self.adapters, open_test_split)

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
    def _registries(
        settings: Settings | None,
        tasks: DownstreamTaskRepository | None,
        campaigns: EvaluationCampaignRepository | None,
    ) -> tuple[DownstreamTaskRepository, EvaluationCampaignRepository]:
        """The two registries, on one engine wherever either was left to the root.

        One engine because they are two aggregates of one schema in one database, and a second
        would open a second pool to the same place.

        Raises:
            ValueError: If one was left to the root without settings to build it from.
        """
        if tasks is not None and campaigns is not None:
            return tasks, campaigns
        engine = configured_engine(settings_for(settings, "the registries"))
        return (
            SqlAlchemyDownstreamTaskRepository(engine) if tasks is None else tasks,
            SqlAlchemyEvaluationCampaignRepository(engine) if campaigns is None else campaigns,
        )

    @staticmethod
    def _candidates(
        store: ArtifactStore,
        tasks: DownstreamTaskRepository,
        workspace: Path,
        corpora: Path,
        backbone: ArtifactRef | None,
        lora: LoraSpec | None,
        schedule: AdaptationSchedule,
        device: str | None,
        open_test_split: OpenTestSplit,
    ) -> CandidateProvider:
        if backbone is None or lora is None:
            raise ValueError(
                "without a backbone and its low-rank updates the process needs its candidates "
                "given, not built"
            )
        blocks = PublishedCorpusBlocks(store, workspace)
        corpus = BlockCorpusWindows(blocks)
        truth = KnownGroundTruths.under(corpora)
        return BackboneCandidateProvider(
            KnownArms.over(backbone, lora),
            schedule,
            RunAdaptation(
                DrawRunLabels(
                    tasks, corpus, truth, DrawLabelBudget(tasks, corpus, truth), open_test_split
                ),
                TorchAdaptationRuntime(
                    RestoredBackbones(store, backbone),
                    blocks,
                    device=device or available_device(),
                    store=store,
                ),
            ),
        )

    @staticmethod
    def _jobs(settings: Settings | None) -> JobQueue:
        broker = settings_for(settings, "jobs").require_broker()
        return CeleryJobQueue(celery_application(broker.url()))

    @staticmethod
    def _services(adapters: Adapters, open_test_split: OpenTestSplit) -> Services:
        complete = CompleteCampaign(
            adapters.campaigns,
            CampaignCompletedAssembler(),
            adapters.clock,
            adapters.ids,
            adapters.events,
        )
        return Services(
            define_campaign=DefineCampaign(
                adapters.tasks,
                adapters.campaigns,
                adapters.candidates,
                adapters.ids,
                adapters.clock,
            ),
            advance_campaign=AdvanceCampaign(adapters.campaigns, adapters.jobs),
            run_campaign_cell=RunCampaignCell(adapters.campaigns, adapters.candidates, complete),
            complete_campaign=complete,
            open_test_split=open_test_split,
        )
