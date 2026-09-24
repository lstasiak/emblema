from dataclasses import dataclass

from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository


@dataclass(frozen=True)
class Adapters:
    """The port implementations a run of an order works on.

    The tasks are a registry of this one run, filled from the order: the machine that runs it
    reaches no database, and the order carries the one task its cells need.
    """

    handoff: CampaignHandoff
    tasks: DownstreamTaskRepository
    candidates: CandidateProvider
