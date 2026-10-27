"""A campaign over events, from a declared design to a published verdict, touching nothing.

The same cycle a worker runs over a quantity or an outcome, over events instead: the design takes
the measure from the task and runs at the one budget of every window, every cell keeps its scores,
the verdict reads how they set the events' windows apart, and the message tells consumers the
event-weighted area, the events found at a share of false alarms and the benchmark's event-wise
score. The candidates answer with scores stated up front, so the areas are known.
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
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.exceptions import ProtocolMismatchError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.queues.immediate_job_queue import ImmediateJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.kernel.compute import ComputeTier
from tests.evaluation.support import BOOTSTRAP, COMPUTE, OPENED_AT, RULES, task, units

CONTROL = CandidateRef("summaries_knn")
CONTENDER = CandidateRef("frozen_knn")
MONTHS = tuple(sorted(units(*(f"m{index}" for index in range(8))), key=str))
# Two events among eight windows, one per unit: the second and the sixth window.
EVENTS = (0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
# The control raises its alarm on every normal window and none of the events; the contender
# scores both events above every normal window, the first above its alarm.
SCORES = {
    CONTROL: (0.3, -0.9, 0.2, 0.4, 0.1, -0.8, 0.5, 0.6),
    CONTENDER: (-0.6, 0.4, -0.5, -0.7, -0.8, -0.1, -0.9, -0.4),
}
EVERYTHING = LabelBudget.everything()
ABSOLUTE = RULES.__class__(
    threshold=ThresholdKind.ABSOLUTE,
    minimum_reduction=0.05,
    floor_part=0.01,
    correction=RULES.correction,
    secondary_family_size=RULES.secondary_family_size,
)


def detector(ref: CandidateRef, kind: CandidateKind = CandidateKind.DETECTOR) -> CampaignCandidate:
    return CampaignCandidate(
        ref=ref,
        kind=kind,
        budget=COMPUTE if kind is CandidateKind.NEURAL else None,
        method=CandidateMethod.of(neighbours=5, contamination=0.02),
        starts_from=None,
    )


class Campaign:
    """The context over adapters that touch nothing, wired as a worker is, over a detection task."""

    def __init__(self, contender_kind: CandidateKind = CandidateKind.DETECTOR) -> None:
        self.published: list[CampaignCompleted] = []
        tasks = InMemoryDownstreamTaskRepository()
        tasks.save(
            task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None)
        )
        self.campaigns = InMemoryEvaluationCampaignRepository()
        candidates = InMemoryCandidateProvider(
            (detector(CONTROL), detector(CONTENDER, contender_kind)),
            MONTHS,
            StatedAnswers(lambda cell: tuple(zip(EVENTS, SCORES[cell.candidate], strict=True))),
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

    def ran(self, budgets: tuple[LabelBudget, ...] = (EVERYTHING,)) -> CampaignId:
        campaign_id = self.define(
            DefineCampaignCommand(
                task=task().task_id,
                purpose=RunPurpose.TUNING,
                tier=ComputeTier.S,
                candidates=(CONTROL, CONTENDER),
                control=CONTROL,
                endpoint=CONTENDER,
                budgets=budgets,
                endpoint_budget=budgets[-1],
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


def test_a_campaign_over_events_is_read_by_how_its_detectors_set_them_apart(
    running: Campaign,
) -> None:
    finished = running.campaigns.read(running.ran())

    assert finished.campaign.design.measure is ErrorMeasure.PR_AUC_SHORTFALL
    assert finished.campaign.design.budgets == (EVERYTHING,)
    assert all(result.predictions for result in finished.results)
    endpoint = finished.verdict().endpoint
    # The control admits both events last: precision 1/7 and 2/8, area 0.196; the contender
    # admits them first.
    assert endpoint.control_error.pooled == pytest.approx(1.0 - (1 / 7 + 2 / 8) / 2)
    assert endpoint.difference.reduction == pytest.approx(1.0 - (1 / 7 + 2 / 8) / 2)
    assert endpoint.verdict is ComparisonVerdict.CONFIRMED


def test_the_message_names_each_detectors_area_events_found_and_event_wise_score(
    running: Campaign,
) -> None:
    running.ran()

    (completed,) = running.published
    control, contender = completed.candidates
    by_name = {m.metric: m.value for m in contender.metrics}
    assert by_name["pr_auc"] == 1.0
    assert by_name["event_recall_at_1pct_false_alarms"] == 1.0
    # The contender's alarm catches the first event alone, on one window, with no false alarm
    # and every normal window quiet: precision one, recall one half.
    assert by_name["event_f0.5"] == pytest.approx(1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5))
    control_by_name = {m.metric: m.value for m in control.metrics}
    assert control_by_name["event_recall_at_1pct_false_alarms"] == 0.0
    assert control_by_name["event_f0.5"] == 0.0
    assert {m.budget for m in contender.metrics} == {None}
    assert "raises the event-weighted PR-AUC" in completed.verdict


def test_a_detection_campaign_over_budgets_of_labels_is_refused(running: Campaign) -> None:
    with pytest.raises(ProtocolMismatchError, match="spread a campaign over"):
        running.ran(budgets=(LabelBudget.of(50), EVERYTHING))


@pytest.mark.parametrize("kind", [CandidateKind.NEURAL, CandidateKind.CLASSICAL])
def test_a_candidate_that_fits_on_labels_is_refused_where_the_task_spends_none(
    kind: CandidateKind,
) -> None:
    with pytest.raises(ProtocolMismatchError, match="fits on labels"):
        Campaign(contender_kind=kind).ran()
