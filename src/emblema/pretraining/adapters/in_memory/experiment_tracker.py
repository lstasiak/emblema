from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome


class InMemoryExperimentTracker:
    """Keeps a run in this process, for a run nobody else has to see.

    What a tracking server would hold, held as attributes: a smoke run, a test and the report all
    want the epochs as they arrive without a server being up. A run that was begun and never
    ended reads as one that was interrupted, here as anywhere else.

    Attributes:
        configuration: What the run was configured to do; ``None`` before it began.
        corpus: The corpus it read.
        run: What the run is called.
        epochs: What each epoch measured, in the order they finished.
        outcome: What the run produced; ``None`` while it is still running or if it never ended.
    """

    def __init__(self) -> None:
        self.configuration: ExperimentConfiguration | None = None
        self.corpus: str | None = None
        self.run: str | None = None
        self.epochs: list[EpochOutcome] = []
        self.outcome: TrainingOutcome | None = None

    def begin(self, configuration: ExperimentConfiguration, *, corpus: str, run: str) -> None:
        if self.configuration is not None:
            raise RuntimeError(f"already recording the run {self.run!r}")
        self.configuration = configuration
        self.corpus = corpus
        self.run = run

    def log_epoch(self, outcome: EpochOutcome) -> None:
        self._recording()
        self.epochs.append(outcome)

    def end(self, outcome: TrainingOutcome) -> None:
        self._recording()
        self.outcome = outcome

    def _recording(self) -> None:
        if self.configuration is None:
            raise RuntimeError("no run has begun")
        if self.outcome is not None:
            raise RuntimeError(f"the run {self.run!r} has ended")
