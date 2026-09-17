import time

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunStatus

from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome


class MlflowExperimentTracker:
    """Records a run against a tracking server, or a directory of files where there is no server.

    The experiment is the configuration's name and each run is one of its runs, so runs of one
    configuration line up and runs of different ones do not. Epochs are logged as they finish,
    which is what makes a curve worth watching while a session that may drop is still running.

    The client is used directly rather than through the module-level API: a global active run is
    process state, and a process that trains two things would then have to remember which of them
    it is logging to.
    """

    def __init__(self, tracking_uri: str) -> None:
        """Record against the server or the directory ``tracking_uri`` names."""
        self._client = MlflowClient(tracking_uri)
        self._run_id: str | None = None
        self._ended = False

    def begin(self, configuration: ExperimentConfiguration, *, corpus: str, run: str) -> None:
        if self._run_id is not None:
            raise RuntimeError(f"already recording the run {self._run_id}")
        experiment = self._client.get_experiment_by_name(configuration.name)
        experiment_id = (
            self._client.create_experiment(configuration.name)
            if experiment is None
            else experiment.experiment_id
        )
        started = self._client.create_run(
            experiment_id, run_name=run, tags={"corpus": corpus, "tier": str(configuration.tier)}
        )
        self._run_id = started.info.run_id
        self._client.log_batch(
            self._run_id,
            params=[Param(key, str(value)) for key, value in configuration.parameters().items()],
        )

    def log_epoch(self, outcome: EpochOutcome) -> None:
        run_id = self._recording()
        stamp = int(time.time() * 1000)
        # A corpus of the mixture gets metrics of its own beside the run's: a single validation
        # curve over several corpora is the curve of whichever holds the largest values.
        per_corpus = [
            (f"validation_loss_{scored.corpus}", scored.loss) for scored in outcome.validation
        ] + [
            (f"relative_validation_{scored.corpus}", scored.relative)
            for scored in outcome.validation
        ]
        self._client.log_batch(
            run_id,
            metrics=[
                Metric(key, value, stamp, outcome.epoch)
                for key, value in [
                    ("training_loss", outcome.training_loss),
                    ("validation_loss", outcome.validation_loss),
                    ("relative_validation", outcome.relative_validation),
                    ("hidden_ratio", outcome.hidden_ratio),
                    ("seconds", outcome.seconds),
                    *per_corpus,
                ]
            ],
        )
        # The latest checkpoint is a tag rather than a parameter: it is what a dropped session is
        # picked up from, and it changes as the run goes on.
        if outcome.checkpoint is not None:
            self._client.set_tag(run_id, "checkpoint", outcome.checkpoint.key)

    def end(self, outcome: TrainingOutcome) -> None:
        run_id = self._recording()
        self._client.set_tag(run_id, "backbone", outcome.backbone.key)
        self._client.set_tag(run_id, "backbone_checksum", str(outcome.backbone.checksum))
        self._client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        self._ended = True

    def _recording(self) -> str:
        if self._run_id is None:
            raise RuntimeError("no run has begun")
        if self._ended:
            raise RuntimeError(f"the run {self._run_id} has ended")
        return self._run_id
