"""A campaign over a binary task, from a declared design to a published verdict, touching nothing.

The same cycle a worker runs over a quantity, over outcomes instead: the design takes the
measure from the task, every cell keeps its answers, the verdict reads how they rank, and the
message tells consumers the area and the Brier score rather than an error in a unit the task
does not have. The candidates answer with probabilities stated up front, so the areas are known.
"""

from collections.abc import Mapping

import pytest

from emblema.evaluation.adapters.in_memory.candidate_provider import (
    InMemoryCandidateProvider,
    StatedAnswers,
)
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
from emblema.evaluation.application.use_cases.record_cell_result import RecordCellResult
from emblema.evaluation.application.use_cases.run_campaign_cell import (
    RunCampaignCell,
    RunCampaignCellCommand,
)
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
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
    OUTCOME,
    OUTCOMES,
    RULES,
    candidate,
    task,
    units,
)

STAYS = tuple(sorted(units(*(f"s{index}" for index in range(8))), key=str))
DIED = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
# The control ranks every death under every survival; the contender ranks them above.
ANSWERS = {
    CONTROL: (0.1, 0.6, 0.7, 0.2, 0.8, 0.5, 0.9, 0.4),
    CONTENDER: (0.9, 0.2, 0.3, 0.8, 0.1, 0.4, 0.2, 0.3),
}
ABSOLUTE = RULES.__class__(
    threshold=ThresholdKind.ABSOLUTE,
    minimum_reduction=0.05,
    floor_part=0.01,
    correction=RULES.correction,
    secondary_family_size=RULES.secondary_family_size,
)


class Campaign:
    """The context over adapters that touch nothing, wired as a worker is, over a binary task."""

    def __init__(self) -> None:
        self.published: list[CampaignCompleted] = []
        tasks = InMemoryDownstreamTaskRepository()
        tasks.save(task(labels=OUTCOME, strata=OUTCOMES))
        self.campaigns = InMemoryEvaluationCampaignRepository()
        candidates = InMemoryCandidateProvider(
            (candidate(CONTROL), candidate(CONTENDER)),
            STAYS,
            StatedAnswers(lambda cell: tuple(zip(DIED, ANSWERS[cell.candidate], strict=True))),
            InMemoryArtifactStore(),
        )
        subscriptions = InMemoryEventSubscriber()
        subscriptions.subscribe(CampaignCompleted, self.published.append)
        events = InMemoryEventPublisher(subscriptions)
        ids, clock = SequentialIdGenerator(), FixedClock(OPENED_AT)
        self.define = DefineCampaign(tasks, self.campaigns, candidates, ids, clock)
        complete = CompleteCampaign(
            self.campaigns, CampaignCompletedAssembler(), clock, ids, events
        )
        self.run_cell = RunCampaignCell(
            self.campaigns, candidates, RecordCellResult(self.campaigns, complete)
        )
        self.advance = AdvanceCampaign(
            self.campaigns, ImmediateJobQueue({RUN_CAMPAIGN_CELL: self._run}), complete
        )

    def ran(self) -> CampaignId:
        campaign_id = self.define(
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
                rules=ABSOLUTE,
                bootstrap=BOOTSTRAP,
            )
        )
        self.advance(AdvanceCampaignCommand(campaign=campaign_id))
        return campaign_id

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


def test_a_campaign_over_outcomes_is_read_by_how_its_candidates_rank_them(
    running: Campaign,
) -> None:
    finished = running.campaigns.read(running.ran())

    assert finished.campaign.design.measure is ErrorMeasure.AUROC_SHORTFALL
    assert all(result.predictions for result in finished.results)
    endpoint = finished.verdict().endpoint
    assert endpoint.difference.reduction == 1.0
    assert endpoint.verdict is ComparisonVerdict.CONFIRMED


def test_the_message_names_each_candidates_area_and_brier_score(running: Campaign) -> None:
    running.ran()

    (completed,) = running.published
    control, contender = completed.candidates
    assert {(m.metric, m.budget, m.value) for m in contender.metrics if m.metric == "auroc"} == {
        ("auroc", 50, 1.0),
        ("auroc", 200, 1.0),
    }
    assert {(m.metric, m.value) for m in control.metrics if m.metric == "auroc"} == {("auroc", 0.0)}
    assert all(0.0 < m.value < 1.0 for m in control.metrics if m.metric == "brier")
    assert "raises the AUROC" in completed.verdict
