"""A campaign is read by the measure its task's target calls for, unless it names one that fits."""

import pytest

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.use_cases.define_campaign import (
    DefineCampaign,
    DefineCampaignCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import MismatchedErrorMeasureError
from emblema.evaluation.domain.labels.label_scheme import LabelScheme
from emblema.evaluation.domain.labels.stratification import Stratification
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.compute import ComputeTier
from tests.evaluation.support import (
    BOOTSTRAP,
    BUDGETS,
    OPENED_AT,
    OUTCOME,
    OUTCOMES,
    RULES,
    SCHEME,
    STRATA,
    convolutions,
    task,
)

TREES, ROCKET = CandidateRef("trees"), CandidateRef("rocket")


def defined(
    labels: LabelScheme, strata: Stratification, measure: ErrorMeasure | None
) -> ErrorMeasure:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task(labels=labels, strata=strata))
    campaigns = InMemoryEvaluationCampaignRepository()
    catalogue = ClassicalBaselineCatalogue(
        tuple(ClassicalArm(ref=ref, method=convolutions(), sources=()) for ref in (TREES, ROCKET))
    )
    define = DefineCampaign(
        tasks, campaigns, catalogue, SequentialIdGenerator(), FixedClock(OPENED_AT)
    )
    campaign_id = define(
        DefineCampaignCommand(
            task=task().task_id,
            purpose=RunPurpose.TUNING,
            tier=ComputeTier.S,
            candidates=(TREES, ROCKET),
            control=TREES,
            endpoint=ROCKET,
            budgets=BUDGETS,
            endpoint_budget=BUDGETS[1],
            seeds=(1,),
            rules=RULES,
            bootstrap=BOOTSTRAP,
            measure=measure,
        )
    )
    return campaigns.get(campaign_id).design.measure


def test_a_campaign_over_a_quantity_is_read_by_squared_error_unless_it_says_otherwise() -> None:
    assert defined(SCHEME, STRATA, None) is ErrorMeasure.RMSE


def test_a_campaign_over_outcomes_is_read_by_how_the_answers_rank_them() -> None:
    assert defined(OUTCOME, OUTCOMES, None) is ErrorMeasure.AUROC_SHORTFALL


def test_a_campaign_that_names_a_measure_its_task_cannot_be_read_by_is_refused() -> None:
    with pytest.raises(MismatchedErrorMeasureError, match="binary target"):
        defined(OUTCOME, OUTCOMES, ErrorMeasure.RMSE)
