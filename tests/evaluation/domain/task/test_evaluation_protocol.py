"""Which question a task asks, and what the answer decides about the task itself."""

import pytest

from emblema.evaluation.domain.exceptions import ProtocolMismatchError
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from tests.evaluation.support import task


def test_only_the_supervised_protocol_spends_labels() -> None:
    assert EvaluationProtocol.LABEL_BUDGET.spends_labels
    assert not EvaluationProtocol.ANOMALY_DETECTION.spends_labels


def test_a_detection_task_that_carries_a_label_scheme_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="carries a label scheme"):
        task(protocol=EvaluationProtocol.ANOMALY_DETECTION)


def test_a_supervised_task_without_a_label_scheme_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="carries no label scheme"):
        task(labels=None, strata=None)


def test_a_supervised_task_without_a_stratification_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="carries no stratification"):
        task(strata=None)


def test_a_detection_task_reads_no_label_and_draws_no_budget() -> None:
    detection = task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=None, strata=None)

    with pytest.raises(ProtocolMismatchError, match="reads no label"):
        detection.label_scheme()
    with pytest.raises(ProtocolMismatchError, match="draws no budget"):
        detection.stratification()


def test_a_detection_task_refuses_a_campaign_that_would_spread_it_over_budgets() -> None:
    detection = task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=None, strata=None)

    with pytest.raises(ProtocolMismatchError, match="spread a campaign over"):
        detection.accept_campaign()


def test_a_supervised_task_accepts_a_campaign() -> None:
    task().accept_campaign()
