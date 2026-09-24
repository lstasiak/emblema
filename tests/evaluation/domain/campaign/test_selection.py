"""A campaign that chooses among variants, and a design that runs what was chosen."""

from dataclasses import replace
from typing import Any

import pytest

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignDesignError,
    InvalidInnerHoldoutError,
    SelectionNotReadableError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from tests.evaluation.support import (
    BUDGETS,
    FINER,
    ROCKET,
    SELECTED_BY,
    baseline,
    campaign,
    cell,
    design,
    selection,
)

AT_200 = LabelBudget.of(200)
OTHER = CandidateRef("other")


def test_a_finished_selection_chooses_the_variant_that_is_clearly_better() -> None:
    assert selection().selected(ROCKET, AT_200) == FINER


def test_a_selection_whose_variant_is_within_noise_keeps_the_default() -> None:
    close = selection({ROCKET: (10.0, 11.0, 9.0), FINER: (9.8, 10.9, 8.9)})

    assert close.selected(ROCKET, AT_200) == ROCKET


def test_a_selection_is_not_read_before_it_has_finished() -> None:
    unfinished = replace(selection(), completed_at=None, results=())

    with pytest.raises(SelectionNotReadableError, match="finished selection"):
        unfinished.selected(ROCKET, AT_200)


def test_a_comparison_is_not_read_as_a_selection() -> None:
    with pytest.raises(SelectionNotReadableError):
        campaign().selected(ROCKET, AT_200)


def test_a_selection_holding_no_variant_of_the_candidate_chooses_nothing() -> None:
    with pytest.raises(SelectionNotReadableError, match="no choice"):
        selection().selected(FINER, AT_200)


def test_a_selection_campaign_divides_the_tuning_side_and_no_other_does() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="selection"):
        replace(selection(), design=replace(selection().design, inner_holdout=None))
    with pytest.raises(InvalidCampaignDesignError, match="selection"):
        replace(campaign(), design=replace(design(), inner_holdout=InnerHoldout(one_in=5)))


def test_a_selection_cell_is_asked_of_its_candidate_with_the_division_of_the_tuning_side() -> None:
    asked = selection().evaluation_of(cell(FINER, AT_200, 2))

    assert asked.holdout == InnerHoldout(one_in=5)
    assert asked.purpose is RunPurpose.SELECTION


def test_a_run_and_its_division_must_go_together() -> None:
    asked = selection().evaluation_of(cell(FINER, AT_200, 2))

    with pytest.raises(InvalidInnerHoldoutError, match="no division"):
        replace(asked, holdout=None)
    with pytest.raises(InvalidInnerHoldoutError, match="outside the tuning side"):
        CandidateEvaluation(
            task=asked.task,
            cell=asked.cell,
            purpose=RunPurpose.TUNING,
            retain=False,
            declared=asked.declared,
            holdout=InnerHoldout(one_in=5),
        )


def tuned_design(**overrides: Any) -> CampaignDesign:
    stated: dict[str, Any] = {
        "candidates": (baseline(ROCKET), baseline(OTHER)),
        "control": OTHER,
        "endpoint": ROCKET,
        "tuned": (
            TunedChoice(candidate=ROCKET, budget=AT_200, variant=FINER, selected_by=SELECTED_BY),
        ),
        "variants": (baseline(FINER, 2.0),),
    }
    return replace(design(), **(stated | overrides))


def test_a_tuned_pairing_runs_its_variant_and_every_other_runs_the_base() -> None:
    tuned = tuned_design()

    assert tuned.declared_for(cell(ROCKET, AT_200, 1)).ref == FINER
    assert tuned.declared_for(cell(ROCKET, BUDGETS[0], 1)).ref == ROCKET
    # The curve still reads the pairing under the candidate's own name.
    assert {c.candidate for c in tuned.cells()} == {ROCKET, OTHER}


def test_a_tuned_variant_that_is_not_described_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="not described"):
        tuned_design(variants=())


def test_a_variant_described_that_nothing_runs_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="no pairing runs"):
        tuned_design(tuned=())


def test_a_variant_of_another_kind_than_its_base_is_refused() -> None:
    neural = replace(
        baseline(FINER, 2.0), kind=design().candidates[0].kind, budget=design().candidates[0].budget
    )

    with pytest.raises(InvalidCampaignDesignError, match="kind and compute budget"):
        tuned_design(variants=(neural,))


def test_a_pairing_tuned_at_a_budget_the_grid_lacks_is_refused() -> None:
    elsewhere = TunedChoice(
        candidate=ROCKET, budget=LabelBudget.of(7), variant=FINER, selected_by=SELECTED_BY
    )

    with pytest.raises(InvalidCampaignDesignError, match="budget the grid lacks"):
        tuned_design(tuned=(elsewhere,))
