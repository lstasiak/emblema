from pathlib import Path

from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_engine, configured_store, settings_for
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.workers.adapters import Adapters
from emblema.entrypoints.workers.services import Services
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.persistence.downstream_task_repository import (
    SqlAlchemyDownstreamTaskRepository,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_campaign_cell import RunCampaignCell
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


class CampaignProcess:
    """Everything a worker of a campaign is made of, apart from the candidates it supplies.

    Two processes work one grid: one carries the machine-learning stack and adapts a backbone,
    the other carries none of it and fits candidates that start from no weights. That is the
    whole of what separates them — the same store, the same two registries, the same queue, the
    same drawing of labels — so the rest is assembled once here and each root is left stating
    the one thing it really is. Written as a class the roots hold rather than inherit: a root is
    a composition, and a composition that inherits its dependencies is no longer readable as one.

    Its lifetimes are all process-scoped, which is a decision rather than a simplification: the
    one thing worth scoping per task would be the backbone, and reading a backbone out of the
    store for each of a grid's cells would download it eighty times to save holding one copy.

    Handlers of events are registered through the subscriber, and use cases publish through the
    publisher, so what listens is decided here and not by whoever publishes. The publisher is
    the synchronous one: nothing yet has to outlive the transaction that publishes it, and an
    outbox is the answer on the day something does.

    Attributes:
        store: Where artifacts are read and kept.
        tasks: Registry of downstream tasks.
        campaigns: Registry of campaigns.
        jobs: Where cells are submitted.
        events: Where use cases publish.
        subscriptions: Where handlers are registered.
        clock: Source of the current instant.
        ids: Source of new identifiers.
        blocks: Published corpora, fetched and mapped — what a runtime reads windows through.
        open_test_split: The operation that records the asking for the frozen side.
        draw_label_budget: One budget of labels, drawn from a task's tuning side.
        draw_run_labels: What a run of any candidate is given before anything is fitted.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        workspace: Path,
        corpora: Path,
        store: ArtifactStore | None = None,
        tasks: DownstreamTaskRepository | None = None,
        campaigns: EvaluationCampaignRepository | None = None,
        jobs: JobQueue | None = None,
        subscriptions: InMemoryEventSubscriber | None = None,
        events: EventPublisher | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble everything but the candidates.

        Args:
            settings: Values read from the environment; only this and the adapters see them.
            workspace: Directory corpus blocks are fetched to and mapped from.
            corpora: Directory the raw corpora sit in, where a task's labels are read from.
            store: Artifact store; the configured S3-compatible bucket unless given.
            tasks: Registry of tasks; the configured metadata database unless given.
            campaigns: Registry of campaigns; the same database unless given.
            jobs: Where cells are submitted; the configured broker unless given.
            subscriptions: Where handlers are registered; a fresh registry unless given.
            events: Where use cases publish; synchronous over ``subscriptions`` unless given.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.

        Raises:
            ValueError: If an adapter is left to be built without settings to build it from.
        """
        self.store = configured_store(settings_for(settings, "store")) if store is None else store
        self.subscriptions = InMemoryEventSubscriber() if subscriptions is None else subscriptions
        self.events = InMemoryEventPublisher(self.subscriptions) if events is None else events
        self.clock = SystemClock() if clock is None else clock
        self.ids = Uuid4IdGenerator() if ids is None else ids
        self.tasks, self.campaigns = self._registries(settings, tasks, campaigns)
        self.jobs = self._jobs(settings) if jobs is None else jobs
        self.open_test_split = OpenTestSplit(self.tasks, self.ids, self.clock, self.events)
        self.blocks = PublishedCorpusBlocks(self.store, workspace)
        corpus = BlockCorpusWindows(self.blocks)
        truth = KnownGroundTruths.under(corpora)
        self.draw_label_budget = DrawLabelBudget(self.tasks, corpus, truth)
        self.draw_run_labels = DrawRunLabels(
            self.tasks, corpus, truth, self.draw_label_budget, self.open_test_split
        )

    def assemble(self, candidates: CandidateProvider) -> tuple[Adapters, Services]:
        """The process as it stands once what it competes is known."""
        adapters = Adapters(
            store=self.store,
            tasks=self.tasks,
            campaigns=self.campaigns,
            candidates=candidates,
            jobs=self.jobs,
            events=self.events,
            subscriptions=self.subscriptions,
            clock=self.clock,
            ids=self.ids,
        )
        complete = CompleteCampaign(
            adapters.campaigns,
            CampaignCompletedAssembler(),
            adapters.clock,
            adapters.ids,
            adapters.events,
        )
        return adapters, Services(
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
            open_test_split=self.open_test_split,
        )

    @staticmethod
    def _registries(
        settings: Settings | None,
        tasks: DownstreamTaskRepository | None,
        campaigns: EvaluationCampaignRepository | None,
    ) -> tuple[DownstreamTaskRepository, EvaluationCampaignRepository]:
        """The two registries, on one engine wherever either was left to be built.

        One engine because they are two aggregates of one schema in one database, and a second
        would open a second pool to the same place.

        Raises:
            ValueError: If one was left to be built without settings to build it from.
        """
        if tasks is not None and campaigns is not None:
            return tasks, campaigns
        engine = configured_engine(settings_for(settings, "the registries"))
        return (
            SqlAlchemyDownstreamTaskRepository(engine) if tasks is None else tasks,
            SqlAlchemyEvaluationCampaignRepository(engine) if campaigns is None else campaigns,
        )

    @staticmethod
    def _jobs(settings: Settings | None) -> JobQueue:
        broker = settings_for(settings, "jobs").require_broker()
        return CeleryJobQueue(celery_application(broker.url()))
