from dataclasses import dataclass

from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.event_subscriber import EventSubscriber
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.job_queue import JobQueue


@dataclass(frozen=True)
class Adapters:
    """Every port implementation the worker runs on, so that what a use case got can be read.

    All of them are process-scoped, and deliberately so: the expensive one is the candidate
    provider, which holds a backbone read out of the store once and reused by every cell.
    """

    store: ArtifactStore
    tasks: DownstreamTaskRepository
    campaigns: EvaluationCampaignRepository
    candidates: CandidateProvider
    jobs: JobQueue
    events: EventPublisher
    subscriptions: EventSubscriber
    clock: Clock
    ids: IdGenerator
