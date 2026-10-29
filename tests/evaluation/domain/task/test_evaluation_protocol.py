"""Which question a task asks, and what the answer decides about the task itself."""

import pytest

from emblema.evaluation.domain.exceptions import ProtocolMismatchError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from tests.evaluation.support import OUTCOME, OUTCOMES, task

DETECTION = EvaluationProtocol.ANOMALY_DETECTION


def test_only_the_supervised_protocol_spends_labels() -> None:
    assert EvaluationProtocol.LABEL_BUDGET.spends_labels
    assert not DETECTION.spends_labels


def test_a_detection_task_that_reads_a_quantity_or_an_outcome_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="does not read its windows by continuous"):
        task(protocol=DETECTION, strata=None)
    with pytest.raises(ProtocolMismatchError, match="does not read its windows by binary"):
        task(protocol=DETECTION, labels=OUTCOME, strata=None)


def test_a_supervised_task_that_reads_events_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="does not read its windows by anomaly"):
        task(labels=AnomalyScheme(), strata=OUTCOMES)


def test_a_supervised_task_without_a_stratification_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="carries no stratification"):
        task(strata=None)


def test_a_detection_task_with_a_stratification_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="carries a stratification"):
        task(protocol=DETECTION, labels=AnomalyScheme(), strata=OUTCOMES)


def test_a_detection_task_reads_events_and_draws_no_budget() -> None:
    detection = task(protocol=DETECTION, labels=AnomalyScheme(), strata=None)

    assert detection.label_scheme() == AnomalyScheme()
    with pytest.raises(ProtocolMismatchError, match="draws no budget"):
        detection.stratification()


def test_a_detection_task_refuses_a_campaign_that_would_spread_it_over_budgets() -> None:
    detection = task(protocol=DETECTION, labels=AnomalyScheme(), strata=None)

    with pytest.raises(ProtocolMismatchError, match="spread a campaign over"):
        detection.accept_campaign((LabelBudget.of(50), LabelBudget.everything()))
    with pytest.raises(ProtocolMismatchError, match="spread a campaign over"):
        detection.accept_campaign((LabelBudget.of(50),))


def test_a_detection_task_accepts_a_campaign_over_every_window_alone() -> None:
    detection = task(protocol=DETECTION, labels=AnomalyScheme(), strata=None)

    detection.accept_campaign((LabelBudget.everything(),))


def test_a_supervised_task_accepts_a_campaign_over_any_budgets() -> None:
    task().accept_campaign((LabelBudget.of(50), LabelBudget.everything()))


def test_a_detection_task_lends_no_labels_and_borrows_none() -> None:
    detection = task(protocol=DETECTION, labels=AnomalyScheme(), strata=None)

    with pytest.raises(ProtocolMismatchError, match="spends no labels"):
        task().accept_source(detection)
    with pytest.raises(ProtocolMismatchError, match="spends no labels"):
        detection.accept_source(task())
