"""What the Evaluation context publishes, and what a malformed message is refused for."""

import pytest

from emblema.evaluation.contracts.candidate_metric import CandidateMetric
from emblema.evaluation.contracts.exceptions import (
    InvalidCandidateMetricError,
    InvalidCandidateRefError,
)
from emblema.evaluation.contracts.identifiers import CandidateRef


@pytest.mark.parametrize("value", ["", " ", " lora", "lora "])
def test_a_candidate_reference_that_is_blank_or_padded_is_refused(value: str) -> None:
    with pytest.raises(InvalidCandidateRefError):
        CandidateRef(value)


def test_a_candidate_reference_reads_as_what_it_is_called() -> None:
    assert str(CandidateRef("lora")) == "lora"


def test_a_metric_of_every_label_the_tuning_side_holds_carries_no_count() -> None:
    stated = CandidateMetric(metric="rmse", budget=None, value=7.2, repeats=5)

    assert stated.budget is None


@pytest.mark.parametrize("named", ["", " rmse"])
def test_a_metric_without_a_readable_name_is_refused(named: str) -> None:
    with pytest.raises(InvalidCandidateMetricError, match="must be named"):
        CandidateMetric(metric=named, budget=200, value=7.2, repeats=5)


def test_a_counted_budget_of_no_window_is_refused() -> None:
    with pytest.raises(InvalidCandidateMetricError, match="at least one window"):
        CandidateMetric(metric="rmse", budget=0, value=7.2, repeats=5)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_a_figure_that_is_not_finite_is_refused(value: float) -> None:
    with pytest.raises(InvalidCandidateMetricError, match="must be finite"):
        CandidateMetric(metric="rmse", budget=200, value=value, repeats=5)


def test_a_metric_standing_on_no_repeat_is_refused() -> None:
    with pytest.raises(InvalidCandidateMetricError, match="at least one repeat"):
        CandidateMetric(metric="rmse", budget=200, value=7.2, repeats=0)
