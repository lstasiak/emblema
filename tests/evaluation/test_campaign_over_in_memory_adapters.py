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

from emblema.evaluation.adapters.in_memory.campaign_handoff import InMemoryCampaignHandoff
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
from emblema.evaluation.application.use_cases.accept_campaign_order_result import (
    AcceptCampaignOrderResult,
    AcceptCampaignOrderResultCommand,
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
from emblema.evaluation.application.use_cases.fulfil_campaign_order import (
    FulfilCampaignOrder,
    FulfilCampaignOrderCommand,
)
from emblema.evaluation.application.use_cases.order_campaign_cells import (
    OrderCampaignCells,
    OrderCampaignCellsCommand,
)
from emblema.evaluation.application.use_cases.record_cell_result import RecordCellResult
from emblema.evaluation.application.use_cases.run_campaign_cell import (
    RunCampaignCell,
    RunCampaignCellCommand,
)
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignOrderRejectedError,
    InvalidCampaignOrderError,
    UnknownCampaignCellError,
)
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.jobs.worker_pool import WorkerPool
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.compute import ComputeTier
from tests.evaluation.support import (
    BOOTSTRAP,
    BUDGETS,
    CONTENDER,
    CONTROL,
    OPENED_AT,
    RULES,
    candidate,
    result,
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
        self.tasks = tasks
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
        self.complete = CompleteCampaign(
            self.campaigns, CampaignCompletedAssembler(), clock, ids, events
        )
        self.record = RecordCellResult(self.campaigns, self.complete)
        self.run_cell = RunCampaignCell(self.campaigns, self.candidates, self.record)
        self.jobs = ImmediateJobQueue({RUN_CAMPAIGN_CELL: self._run})
        self.advance = AdvanceCampaign(self.campaigns, self.jobs, self.complete)

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
        self.run_cell(
            RunCampaignCellCommand(
                campaign=CampaignId.parse(str(arguments["campaign"])),
                cell=CampaignCell(
                    candidate=CandidateRef(str(arguments["candidate"])),
                    budget=LabelBudget.parse(str(arguments["budget"])),
                    seed=int(str(arguments["seed"])),
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


def test_a_cell_outside_the_grid_is_refused_rather_than_recorded(running: Campaign) -> None:
    # The queue may deliver anything; what the campaign answers for is its own design.
    declared = running.declared()

    with pytest.raises(UnknownCampaignCellError, match="is not a cell of campaign"):
        running.run_cell(
            RunCampaignCellCommand(
                campaign=declared,
                cell=CampaignCell(candidate=CONTENDER, budget=LabelBudget.of(7), seed=1),
            )
        )

    assert running.campaigns.get(declared).results == ()


class ACompetitor:
    """The repository with another worker of the same grid interleaved into the first write.

    A second process that read the same campaign, ran a cell of its own and recorded it while
    this one was busy. Nothing here is a fake of the port: the campaigns are kept by the real
    in-memory adapter, and what is simulated is only the moment the other worker got in.
    """

    def __init__(
        self, campaigns: InMemoryEvaluationCampaignRepository, competing: CellResult
    ) -> None:
        self._campaigns = campaigns
        self._competing = competing
        self._interleaved = False

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        return self._campaigns.get(campaign_id)

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        if not self._interleaved:
            self._interleaved = True
            stood = self._campaigns.get(campaign.campaign_id)
            self._campaigns.save(stood.record(self._competing), seen=stood.revision)
        self._campaigns.save(campaign, seen=seen)


def test_a_cell_recorded_while_this_one_ran_is_not_written_over(running: Campaign) -> None:
    campaign_id = running.declared()
    mine, theirs = running.campaigns.get(campaign_id).design.cells()[:2]
    competitor = ACompetitor(
        running.campaigns,
        result(theirs.candidate, theirs.budget, theirs.seed, ERRORS[theirs.candidate]),
    )
    run_cell = RunCampaignCell(
        competitor, running.candidates, RecordCellResult(competitor, running.complete)
    )

    run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=mine))

    assert {stored.cell for stored in running.campaigns.get(campaign_id).results} == {mine, theirs}


def test_the_worker_that_records_last_is_the_one_that_closes_the_grid(running: Campaign) -> None:
    campaign_id = running.declared()
    cells = running.campaigns.get(campaign_id).design.cells()
    for cell in cells[:-2]:
        running.run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=cell))
    mine, theirs = cells[-2], cells[-1]
    competitor = ACompetitor(
        running.campaigns,
        result(theirs.candidate, theirs.budget, theirs.seed, ERRORS[theirs.candidate]),
    )
    run_cell = RunCampaignCell(
        competitor, running.candidates, RecordCellResult(competitor, running.complete)
    )

    # This worker is overtaken on the last two cells, so the state it started from never shows a
    # whole grid; the state it ends up writing does, and that is what has to close the campaign.
    run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=mine))

    assert running.campaigns.get(campaign_id).is_finished
    assert len(running.published) == 1


def test_a_grid_left_whole_but_open_is_closed_by_advancing_it(running: Campaign) -> None:
    # Recording a cell and closing the grid are two writes, so a worker lost between them leaves
    # a campaign with nothing to submit and no cell to run again — one nothing else would close.
    campaign_id = running.declared()
    stood = running.campaigns.get(campaign_id)
    for cell in stood.design.cells():
        recorded = stood.record(
            result(cell.candidate, cell.budget, cell.seed, ERRORS[cell.candidate])
        )
        running.campaigns.save(recorded, seen=stood.revision)
        stood = recorded

    submitted = running.advance(AdvanceCampaignCommand(campaign=campaign_id))

    assert submitted == 0
    assert running.campaigns.get(campaign_id).is_finished
    assert len(running.published) == 1


class AlwaysOvertaken:
    """A repository that is always one step ahead of whoever tries to write to it."""

    def __init__(self, campaigns: InMemoryEvaluationCampaignRepository) -> None:
        self._campaigns = campaigns

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        return self._campaigns.get(campaign_id)

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        raise CampaignChangedElsewhereError("something got there first, and keeps getting there")


def test_a_worker_that_keeps_losing_the_race_gives_up_rather_than_spinning(
    running: Campaign,
) -> None:
    # A store that refuses every write is a failure, not a race, and a worker that treated the
    # two the same would sit in the queue repeating an expensive cell for as long as it lasted.
    campaign_id = running.declared()
    cell = running.campaigns.get(campaign_id).design.cells()[0]
    overtaken = AlwaysOvertaken(running.campaigns)
    run_cell = RunCampaignCell(
        overtaken, running.candidates, RecordCellResult(overtaken, running.complete)
    )

    with pytest.raises(CampaignChangedElsewhereError, match="every one of"):
        run_cell(RunCampaignCellCommand(campaign=campaign_id, cell=cell))

    assert running.campaigns.get(campaign_id).results == ()


class ElsewhereRun:
    """What another machine holds to run an order: the same candidates, no registry of its own."""

    def __init__(self, running: Campaign, handoff: InMemoryCampaignHandoff) -> None:
        self.tasks = InMemoryDownstreamTaskRepository()
        self.fulfil = FulfilCampaignOrder(handoff, self.tasks, running.candidates)


def ordered(
    running: Campaign, handoff: InMemoryCampaignHandoff, campaign_id: CampaignId
) -> ArtifactRef:
    order = OrderCampaignCells(running.campaigns, running.tasks, handoff)
    return order(
        OrderCampaignCellsCommand(campaign=campaign_id, pool=WorkerPool.ML, git_commit="abc123")
    )


def test_a_grid_run_through_an_order_is_the_grid_run_through_the_queue(running: Campaign) -> None:
    queued = running.declared()
    running.advance(AdvanceCampaignCommand(campaign=queued))
    handoff = InMemoryCampaignHandoff()
    through_order = running.declared()
    elsewhere = ElsewhereRun(running, handoff)
    accept = AcceptCampaignOrderResult(handoff, running.record)

    reported = elsewhere.fulfil(
        FulfilCampaignOrderCommand(
            order=ordered(running, handoff, through_order), git_commit="abc123"
        )
    )
    accepted = accept(AcceptCampaignOrderResultCommand(result=reported))

    by_queue = running.campaigns.get(queued)
    by_order = running.campaigns.get(through_order)
    assert accepted == len(by_order.design.cells())
    assert by_order.is_finished
    assert {(r.cell, r.errors) for r in by_order.results} == {
        (r.cell, r.errors) for r in by_queue.results
    }
    assert len(running.published) == 2


def test_every_cell_is_reported_as_it_is_answered(running: Campaign) -> None:
    campaign_id = running.declared()
    handoff = InMemoryCampaignHandoff()
    order = ordered(running, handoff, campaign_id)

    ElsewhereRun(running, handoff).fulfil(
        FulfilCampaignOrderCommand(order=order, git_commit="abc123")
    )

    answered = [len(handoff.read_result(ref).results) for ref in handoff.reported]
    cells = len(running.campaigns.get(campaign_id).design.cells())
    # One report per cell, then the whole once more, which names what the last one did.
    assert answered == [*range(1, cells + 1), cells]
    assert handoff.reported[-1] == handoff.reported[-2]


def test_a_run_resumed_from_its_last_report_runs_only_the_cells_that_are_left(
    running: Campaign,
) -> None:
    campaign_id = running.declared()
    handoff = InMemoryCampaignHandoff()
    order = ordered(running, handoff, campaign_id)
    first = ElsewhereRun(running, handoff)
    first.fulfil(FulfilCampaignOrderCommand(order=order, git_commit="abc123"))
    interrupted = handoff.reported[1]
    asked: list[CampaignCell] = []
    counting = FulfilCampaignOrder(
        handoff, InMemoryDownstreamTaskRepository(), Counting(running.candidates, asked)
    )

    resumed = counting(
        FulfilCampaignOrderCommand(order=order, git_commit="abc123", resume=interrupted)
    )

    cells = running.campaigns.get(campaign_id).design.cells()
    assert len(asked) == len(cells) - 2
    assert handoff.read_result(resumed).cells == frozenset(cells)


def test_an_order_is_refused_on_other_code_before_a_cell_runs(running: Campaign) -> None:
    handoff = InMemoryCampaignHandoff()
    order = ordered(running, handoff, running.declared())

    with pytest.raises(CampaignOrderRejectedError, match="commit"):
        ElsewhereRun(running, handoff).fulfil(
            FulfilCampaignOrderCommand(order=order, git_commit="def456")
        )

    assert handoff.reported == []


def test_an_order_of_a_pool_that_has_no_outstanding_cell_is_refused(running: Campaign) -> None:
    order = OrderCampaignCells(running.campaigns, running.tasks, InMemoryCampaignHandoff())

    with pytest.raises(InvalidCampaignOrderError, match="names no cell"):
        order(
            OrderCampaignCellsCommand(
                campaign=running.declared(), pool=WorkerPool.GENERAL, git_commit="abc123"
            )
        )


def test_a_result_accepted_twice_keeps_the_answers_that_stood_first(running: Campaign) -> None:
    campaign_id = running.declared()
    handoff = InMemoryCampaignHandoff()
    reported = ElsewhereRun(running, handoff).fulfil(
        FulfilCampaignOrderCommand(
            order=ordered(running, handoff, campaign_id), git_commit="abc123"
        )
    )
    accept = AcceptCampaignOrderResult(handoff, running.record)
    accept(AcceptCampaignOrderResultCommand(result=reported))
    once = running.campaigns.get(campaign_id)

    accept(AcceptCampaignOrderResultCommand(result=reported))

    assert running.campaigns.get(campaign_id) == once
    assert len(running.published) == 1


def test_a_result_answering_a_cell_its_order_never_held_is_refused_whole(
    running: Campaign,
) -> None:
    campaign_id = running.declared()
    handoff = InMemoryCampaignHandoff()
    order = ordered(running, handoff, campaign_id)
    stranger = result(CONTROL, LabelBudget.of(50), 99, ERRORS[CONTROL])
    forged = handoff.report(
        CampaignOrderResult(
            order=order, campaign=campaign_id, git_commit="abc123", results=(stranger,)
        )
    )

    with pytest.raises(CampaignOrderRejectedError, match="never held"):
        AcceptCampaignOrderResult(handoff, running.record)(
            AcceptCampaignOrderResultCommand(result=forged)
        )

    assert running.campaigns.get(campaign_id).results == ()


class Counting:
    """A provider that remembers which cells it was asked for, then asks the real one."""

    def __init__(self, provider: InMemoryCandidateProvider, asked: list[CampaignCell]) -> None:
        self._provider = provider
        self._asked = asked

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._provider.describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        self._asked.append(request.cell)
        return self._provider.evaluate(request)
