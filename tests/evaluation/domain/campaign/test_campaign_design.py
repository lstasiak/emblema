from dataclasses import replace

import pytest

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignDesignError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from tests.evaluation.support import (
    CONTENDER,
    CONTROL,
    PROBE,
    RULES,
    candidate,
    design,
)


def test_the_grid_is_every_candidate_at_every_budget_under_every_seed() -> None:
    cells = design().cells()

    assert len(cells) == 2 * 2 * 2
    assert len(set(cells)) == len(cells)
    assert cells[0] == CampaignCell(candidate=CONTROL, budget=LabelBudget.of(50), seed=1)


def test_the_grid_is_laid_out_in_a_fixed_order() -> None:
    assert design().cells() == design().cells()


def test_a_campaign_needs_a_control_and_someone_to_compare_with_it() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="at least one other candidate"):
        design(candidates=(candidate(CONTROL),))


def test_a_candidate_named_twice_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="named twice"):
        design(candidates=(candidate(CONTROL), candidate(CONTENDER), candidate(CONTENDER)))


@pytest.mark.parametrize("role", ["control", "endpoint"])
def test_a_control_or_endpoint_that_does_not_compete_is_refused(role: str) -> None:
    with pytest.raises(InvalidCampaignDesignError, match=f"the {role}"):
        design(**{role: CandidateRef("absent")})


def test_an_endpoint_that_is_the_control_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="is the control"):
        design(endpoint=CONTROL)


@pytest.mark.parametrize("axis", ["budgets", "seeds"])
def test_an_empty_axis_is_refused(axis: str) -> None:
    with pytest.raises(InvalidCampaignDesignError, match="at least one"):
        design(**{axis: ()})


@pytest.mark.parametrize(
    ("axis", "repeated"),
    [("budgets", (LabelBudget.of(50), LabelBudget.of(50))), ("seeds", (1, 1))],
)
def test_a_repeated_coordinate_is_refused(axis: str, repeated: tuple[object, ...]) -> None:
    with pytest.raises(InvalidCampaignDesignError, match="named twice"):
        design(**{axis: repeated})


def test_an_endpoint_at_a_budget_the_campaign_never_runs_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="not one the campaign runs"):
        design(endpoint_budget=LabelBudget.of(1000))


def test_neural_candidates_that_declare_different_compute_budgets_are_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="different ones"):
        design(
            candidates=(
                candidate(CONTROL),
                candidate(CONTENDER, budget=ComputeBudget(epochs=9, min_steps=0, batch_size=2)),
            )
        )


def test_a_classical_candidate_is_not_held_to_the_shared_compute_budget() -> None:
    stated = design(
        candidates=(
            candidate(CONTROL),
            candidate(CONTENDER),
            candidate(PROBE, kind=CandidateKind.CLASSICAL, budget=None),
        )
    )

    assert len(stated.contenders) == 2


def test_a_grid_with_more_secondary_comparisons_than_the_registration_names_is_refused() -> None:
    with pytest.raises(InvalidCampaignDesignError, match="more than the"):
        design(
            candidates=(candidate(CONTROL), candidate(CONTENDER), candidate(PROBE)),
            rules=replace(RULES, secondary_family_size=1),
        )


def test_the_artifact_is_kept_at_the_endpoint_budget_under_the_first_seed() -> None:
    stated = design()

    assert stated.retains(CampaignCell(candidate=CONTROL, budget=LabelBudget.of(200), seed=1))
    assert not stated.retains(CampaignCell(candidate=CONTROL, budget=LabelBudget.of(200), seed=2))
    assert not stated.retains(CampaignCell(candidate=CONTROL, budget=LabelBudget.of(50), seed=1))


def test_a_candidate_the_design_does_not_name_is_refused() -> None:
    with pytest.raises(UnknownCandidateError):
        design().get_candidate(CandidateRef("absent"))
