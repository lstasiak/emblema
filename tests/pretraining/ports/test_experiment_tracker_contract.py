"""Contract of the ExperimentTracker port, run against every adapter.

The port is written to, never read from, so what it owes its caller is the order: a run begins
once, gains epochs while it runs, ends once, and refuses anything after that. What each adapter
stored is its own test's business.
"""

from collections.abc import Callable

import pytest

from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from tests.support.experiments import WEIGHTS, configuration
from tests.support.experiments import epoch_outcome as epoch

ADAPTERS: dict[str, Callable[[str], ExperimentTracker]] = {
    "in-memory": lambda _: InMemoryExperimentTracker(),
    "mlflow": MlflowExperimentTracker,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def tracker(request: pytest.FixtureRequest, tracking_uri: str) -> ExperimentTracker:
    build: Callable[[str], ExperimentTracker] = request.param
    return build(tracking_uri)


def outcome() -> TrainingOutcome:
    return TrainingOutcome(backbone=WEIGHTS, epochs=(epoch(0), epoch(1, backbone=WEIGHTS)))


def test_a_run_is_begun_logged_and_ended(tracker: ExperimentTracker) -> None:
    tracker.begin(configuration(), corpus="control-a", run="first")
    tracker.log_epoch(epoch(0))
    tracker.log_epoch(epoch(1, backbone=WEIGHTS))
    tracker.end(outcome())


def test_an_epoch_before_a_run_has_begun_is_refused(tracker: ExperimentTracker) -> None:
    with pytest.raises(RuntimeError, match="no run has begun"):
        tracker.log_epoch(epoch(0))


def test_an_ending_before_a_run_has_begun_is_refused(tracker: ExperimentTracker) -> None:
    with pytest.raises(RuntimeError, match="no run has begun"):
        tracker.end(outcome())


def test_a_second_run_on_one_tracker_is_refused(tracker: ExperimentTracker) -> None:
    tracker.begin(configuration(), corpus="control-a", run="first")

    with pytest.raises(RuntimeError, match="already"):
        tracker.begin(configuration(), corpus="control-a", run="second")


def test_an_epoch_after_the_run_has_ended_is_refused(tracker: ExperimentTracker) -> None:
    tracker.begin(configuration(), corpus="control-a", run="first")
    tracker.end(outcome())

    with pytest.raises(RuntimeError, match="has ended"):
        tracker.log_epoch(epoch(2))


def test_a_second_ending_is_refused(tracker: ExperimentTracker) -> None:
    tracker.begin(configuration(), corpus="control-a", run="first")
    tracker.end(outcome())

    with pytest.raises(RuntimeError, match="has ended"):
        tracker.end(outcome())
