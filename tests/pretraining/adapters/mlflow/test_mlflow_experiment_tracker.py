"""What MLflow holds after a run: the parameters, the curve, and how the run ended.

Against a database file rather than a server — the same client and the same store the served one
runs on, without the network. That the calls come in the right order is the port's contract; what
they left behind is here.
"""

from mlflow import MlflowClient

from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from tests.support.experiments import WEIGHTS, configuration, epoch_outcome


def test_a_finished_run_holds_its_parameters_its_curve_and_what_it_produced(
    tracking_uri: str,
) -> None:
    uri = tracking_uri
    stated = configuration(name="finished-run")
    epochs = (
        epoch_outcome(0, checkpoint=WEIGHTS),
        epoch_outcome(1, backbone=WEIGHTS),
    )
    tracker = MlflowExperimentTracker(uri)

    tracker.begin(stated, corpus="control-a", run="only")
    for epoch in epochs:
        tracker.log_epoch(epoch)
    tracker.end(TrainingOutcome(backbone=WEIGHTS, epochs=epochs))

    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(stated.name)
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.params["seed"] == "1"
    assert run.data.params["precision"] == "fp32"
    assert run.data.tags["corpus"] == "control-a"
    assert run.data.tags["tier"] == "S"
    assert run.data.tags["backbone"] == WEIGHTS.key
    assert run.info.status == "FINISHED"
    curve = client.get_metric_history(run.info.run_id, "validation_loss")
    assert [(point.step, point.value) for point in curve] == [
        (0, epochs[0].validation_loss),
        (1, epochs[1].validation_loss),
    ]


def test_the_checkpoint_a_dropped_session_would_resume_from_is_the_latest_one(
    tracking_uri: str,
) -> None:
    uri = tracking_uri
    stated = configuration(name="dropped-run")
    tracker = MlflowExperimentTracker(uri)
    tracker.begin(stated, corpus="control-a", run="dropped")

    tracker.log_epoch(epoch_outcome(0, checkpoint=WEIGHTS))
    tracker.log_epoch(epoch_outcome(1))

    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(stated.name)
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    # An epoch that wrote none leaves the last one standing: it is still where a resume starts.
    assert run.data.tags["checkpoint"] == WEIGHTS.key
    assert run.info.status == "RUNNING"
