"""A whole campaign, from a declared design to a published verdict, on adapters that touch nothing.

The cycle a worker runs, with the queue running each job where it was submitted and the
candidates answering with errors stated up front: the grid expands, every cell is submitted and
run, the last one closes the campaign, and the outcome leaves the context as one message. It is
here rather than in a use case test because none of the parts is the point — what is held is
that they compose into a campaign that finishes, and that a dropped run resumes from what was
recorded rather than from the start.
"""

import time
from collections.abc import Mapping

import pytest

from emblema.evaluation.adapters.in_memory.candidate_provider import InMemoryCandidateProvider
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.advance_campaign import (
    RUN_CAMPAIGN_CELL,
    AdvanceCampaign,
    AdvanceCampaignCommand,
)
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import (
    DefineCampaign,
    DefineCampaignCommand,
)
from emblema.evaluation.application.use_cases.run_campaign_cell import (
    RunCampaignCell,
    RunCampaignCellCommand,
)
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.kernel.compute import ComputeTier
from tests.evaluation.support import (
    BOOTSTRAP,
    BUDGETS,
    CONTENDER,
    CONTROL,
    OPENED_AT,
    RULES,
    candidate,
    task,
    units,
)

# The contender errs less on every unit at every budget: a campaign whose answer is known, so
# what is held is the cycle rather than the arithmetic, which the statistics tests cover.
ERRORS = {CONTROL: (6.0, 8.0, 10.0), CONTENDER: (3.0, 4.0, 5.0)}
UNITS = tuple(sorted(units("c0", "c1", "c2"), key=str))


class Campaign:
    """The whole context assembled over adapters that touch nothing, wired as a worker is."""

    def __init__(self) -> None:
        self.published: list[CampaignCompleted] = []
        self.store = InMemoryArtifactStore()
        tasks = InMemoryDownstreamTaskRepository()
        tasks.save(task())
        self.campaigns = InMemoryEvaluationCampaignRepository()
        self.candidates = InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(CONTENDER)),
            UNITS,
            lambda cell: ERRORS[cell.candidate],
            self.store,
        )
        subscriptions = InMemoryEventSubscriber()
        subscriptions.subscribe(CampaignCompleted, self.published.append)
        events = InMemoryEventPublisher(subscriptions)
        ids, clock = SequentialIdGenerator(), FixedClock(OPENED_AT)
        self.define = DefineCampaign(tasks, self.campaigns, self.candidates, ids, clock)
        self.run_cell = RunCampaignCell(
            self.campaigns,
            self.candidates,
            CompleteCampaign(self.campaigns, CampaignCompletedAssembler(), clock, ids, events),
        )
        self.jobs = ImmediateJobQueue({RUN_CAMPAIGN_CELL: self._run})
        self.advance = AdvanceCampaign(self.campaigns, self.jobs)

    def declared(self) -> CampaignId:
        return self.define(
            DefineCampaignCommand(
                task=task().task_id,
                purpose=RunPurpose.TUNING,
                tier=ComputeTier.S,
                candidates=(CONTROL, CONTENDER),
                control=CONTROL,
                endpoint=CONTENDER,
                budgets=BUDGETS,
                endpoint_budget=LabelBudget.of(200),
                seeds=(1, 2),
                rules=RULES,
                bootstrap=BOOTSTRAP,
            )
        )

    def _run(self, arguments: Mapping[str, JobArgument]) -> None:
        budget, seed = arguments["budget"], arguments["seed"]
        self.run_cell(
            RunCampaignCellCommand(
                campaign=CampaignId.parse(str(arguments["campaign"])),
                cell=CampaignCell(
                    candidate=CandidateRef(str(arguments["candidate"])),
                    budget=LabelBudget.everything()
                    if budget is None
                    else LabelBudget.of(int(str(budget))),
                    seed=int(str(seed)),
                ),
            )
        )


@pytest.fixture
def running() -> Campaign:
    return Campaign()


def test_a_declared_campaign_runs_to_a_verdict_and_publishes_it(running: Campaign) -> None:
    campaign_id = running.declared()

    submitted = running.advance(AdvanceCampaignCommand(campaign=campaign_id))

    assert submitted == 2 * len(BUDGETS) * 2
    assert running.campaigns.get(campaign_id).is_finished
    assert len(running.published) == 1


def test_the_outcome_names_every_candidate_what_it_scored_and_where_it_ended(
    running: Campaign,
) -> None:
    campaign_id = running.declared()
    running.advance(AdvanceCampaignCommand(campaign=campaign_id))

    completed = running.published[0]

    assert completed.campaign == campaign_id
    assert [c.candidate for c in completed.candidates] == [CONTROL, CONTENDER]
    assert [c.standing for c in completed.candidates] == [
        CandidateStanding.CONTROL,
        CandidateStanding.ESTABLISHED,
    ]
    assert [m.budget for m in completed.candidates[0].metrics] == [50, 200]


def test_the_outcome_names_the_artifact_of_the_cell_the_design_kept(running: Campaign) -> None:
    running.advance(AdvanceCampaignCommand(campaign=running.declared()))

    kept = [c.artifact for c in running.published[0].candidates]

    assert all(ref is not None for ref in kept)
    assert all(running.store.get(ref) for ref in kept if ref is not None)


def test_a_campaign_resumes_from_what_was_recorded_rather_than_from_the_start(
    running: Campaign,
) -> None:
    campaign_id = running.declared()
    first = running.campaigns.get(campaign_id).design.cells()[0]
    running.run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=first))

    submitted = running.advance(AdvanceCampaignCommand(campaign=campaign_id))

    assert submitted == 2 * len(BUDGETS) * 2 - 1
    assert running.campaigns.get(campaign_id).is_finished


def test_a_cell_delivered_twice_is_answered_from_what_was_recorded(running: Campaign) -> None:
    campaign_id = running.declared()
    first = running.campaigns.get(campaign_id).design.cells()[0]
    once = running.run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=first))

    again = running.run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=first))

    assert again == once
    assert len(running.campaigns.get(campaign_id).results) == 1


def test_the_whole_cycle_costs_less_than_a_second(running: Campaign) -> None:
    started = time.perf_counter()
    running.advance(AdvanceCampaignCommand(campaign=running.declared()))

    assert time.perf_counter() - started < 1.0


def test_the_outcome_carries_what_serving_needs_and_nothing_more(running: Campaign) -> None:
    running.advance(AdvanceCampaignCommand(campaign=running.declared()))

    completed = running.published[0]

    assert set(vars(completed)) == {
        "event_id",
        "occurred_at",
        "campaign",
        "task",
        "verdict",
        "candidates",
    }
    assert set(vars(completed.candidates[0])) == {
        "candidate",
        "kind",
        "artifact",
        "standing",
        "metrics",
    }
