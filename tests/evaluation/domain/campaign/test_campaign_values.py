"""The small values a campaign is stated and recorded in, and what each refuses."""

import pytest

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignCandidateError,
    InvalidCellResultError,
    InvalidComputeBudgetError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.unit_error import UnitError
from tests.evaluation.support import CONTENDER, CONTROL, candidate, cell, result


@pytest.mark.parametrize(("field", "value"), [("epochs", 0), ("batch_size", 0), ("min_steps", -1)])
def test_a_compute_budget_that_buys_nothing_is_refused(field: str, value: int) -> None:
    stated = {"epochs": 2, "min_steps": 0, "batch_size": 2} | {field: value}

    with pytest.raises(InvalidComputeBudgetError):
        ComputeBudget(**stated)


def test_two_candidates_spend_one_budget_when_their_budgets_are_equal() -> None:
    assert ComputeBudget(epochs=2, min_steps=0, batch_size=2) == ComputeBudget(
        epochs=2, min_steps=0, batch_size=2
    )


def test_a_neural_candidate_without_a_compute_budget_is_refused() -> None:
    with pytest.raises(InvalidCampaignCandidateError, match="declares none"):
        candidate(CONTENDER, budget=None)


def test_a_classical_candidate_that_declares_a_shared_budget_is_refused() -> None:
    with pytest.raises(InvalidCampaignCandidateError, match="declares one"):
        candidate(CONTENDER, kind=CandidateKind.CLASSICAL)


def test_the_control_arm_starts_from_no_weights() -> None:
    assert candidate(CONTROL).starts_from is None


def test_a_result_that_scores_no_unit_is_refused() -> None:
    with pytest.raises(InvalidCellResultError, match="scores no unit"):
        CellResult(
            cell=cell(CONTENDER, LabelBudget.of(50), 1), errors=(), seconds=0.0, artifact=None
        )


def test_a_result_that_scores_a_unit_twice_is_refused() -> None:
    twice = UnitError(unit=UnitKey("c"), squared_error=1.0, windows=1)

    with pytest.raises(InvalidCellResultError, match="a unit twice"):
        CellResult(
            cell=cell(CONTENDER, LabelBudget.of(50), 1),
            errors=(twice, twice),
            seconds=0.0,
            artifact=None,
        )


@pytest.mark.parametrize("seconds", [-1.0, float("nan")])
def test_a_time_taken_that_is_not_finite_and_not_negative_is_refused(seconds: float) -> None:
    with pytest.raises(InvalidCellResultError, match="seconds"):
        result(CONTENDER, LabelBudget.of(50), 1, (2.0,), seconds=seconds)


def test_the_error_of_a_cell_is_the_root_mean_over_every_window_it_answered() -> None:
    scored = result(CONTENDER, LabelBudget.of(50), 1, (3.0, 4.0))

    assert scored.rmse == pytest.approx((9.0 + 16.0) ** 0.5 / 2**0.5)
