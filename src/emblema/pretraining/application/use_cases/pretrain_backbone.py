from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class PretrainBackboneCommand:
    """One run: what to do, what to do it on, and where to pick it up from.

    Attributes:
        configuration: The experiment, which fixes everything the run does.
        corpus: The windows to train on and the windows to score on.
        run: What this run is called within the experiment; two runs of one configuration differ
            only here, so the name comes from whoever started them rather than from a clock the
            use case reaches for.
        resume_from: Checkpoint of an interrupted run of the same configuration and corpus.
    """

    configuration: ExperimentConfiguration
    corpus: TrainingCorpus
    run: str
    resume_from: ArtifactRef | None = None


class PretrainBackbone:
    """Trains a backbone under the self-supervised objective and records the run as it happens.

    The use case owns the order of the two ports: the tracker is told what the run is before it
    starts, each epoch as it finishes, and the outcome once there is one. A run that dies halfway
    therefore leaves a tracked run with the epochs it managed and no ending, which is what an
    interrupted run is; nothing is written after the fact.
    """

    def __init__(self, runtime: TrainingRuntime, tracker: ExperimentTracker) -> None:
        self._runtime = runtime
        self._tracker = tracker

    def __call__(self, command: PretrainBackboneCommand) -> TrainingOutcome:
        """Run the experiment and return what it produced.

        Raises:
            InvalidTrainingOutcomeError: If the runtime reported no epoch, or reported the
                backbone anywhere but on the last one.
            IncompatibleCheckpointError: If the checkpoint to resume from belongs elsewhere.
            UnsupportedPrecisionError: If the runtime cannot honour the declared precision.
        """
        # The run is asked for before it is recorded: a runtime refuses a precision it cannot
        # honour or a checkpoint from elsewhere when it is asked, and a run that never started is
        # not a run the tracker should be holding open.
        running = self._runtime.train(command.configuration, command.corpus, command.resume_from)
        self._tracker.begin(command.configuration, corpus=command.corpus.name, run=command.run)
        epochs = []
        for epoch in running:
            self._tracker.log_epoch(epoch)
            epochs.append(epoch)
        # A runtime that yields nothing, or a last epoch without weights, has broken the port's
        # contract: said here rather than by handing the outcome a reference that is not there.
        kept = epochs[-1].backbone if epochs else None
        if kept is None:
            raise InvalidTrainingOutcomeError("the run ended without reporting a backbone")
        outcome = TrainingOutcome(backbone=kept, epochs=tuple(epochs))
        self._tracker.end(outcome)
        return outcome
