"""What a finished campaign tells other contexts about each candidate, by the task's measure."""

from dataclasses import replace
from uuid import UUID

import pytest

from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.shared.events.domain_event import EventId
from tests.evaluation.support import (
    CLOSED_AT,
    CONTENDER,
    CONTROL,
    answered,
    closed_reading,
    design,
    reading,
)

EVENT = EventId(UUID(int=7))
CONTROL_BUDGET = LabelBudget.of(200)


def binary() -> CampaignReading:
    grid = reading(design=replace(design(), measure=ErrorMeasure.AUROC_SHORTFALL))
    answers = {CONTROL: (0.6, 0.5, 0.2, 0.3, 0.1, 0.4), CONTENDER: (0.9, 0.2, 0.8, 0.3, 0.1, 0.4)}
    for run in grid.campaign.design.cells():
        grid = grid.record(answered(run.candidate, run.budget, run.seed, answers[run.candidate]))
    return grid.complete(CLOSED_AT)


def test_a_campaign_over_a_quantity_publishes_its_squared_error() -> None:
    closed = closed_reading()
    message = CampaignCompletedAssembler().assemble(
        closed, closed.verdict(), event_id=EVENT, occurred_at=CLOSED_AT
    )

    assert {metric.metric for c in message.candidates for metric in c.metrics} == {"rmse"}


def test_a_campaign_over_outcomes_publishes_the_area_and_the_brier_score_of_each_candidate() -> (
    None
):
    closed = binary()
    message = CampaignCompletedAssembler().assemble(
        closed, closed.verdict(), event_id=EVENT, occurred_at=CLOSED_AT
    )

    control = next(c for c in message.candidates if c.candidate == CONTROL)
    at_200 = {metric.metric: metric.value for metric in control.metrics if metric.budget == 200}
    assert at_200["auroc"] == pytest.approx(5.0 / 8.0)
    assert at_200["brier"] == pytest.approx(closed.mean_squared_error_of(CONTROL, CONTROL_BUDGET))
