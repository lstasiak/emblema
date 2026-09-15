from typing import Protocol

from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome


class ExperimentTracker(Protocol):
    """Records what a run was configured to do and what it measured, while it runs.

    Three calls in order — the configuration once, each epoch as it finishes, the outcome at the
    end — because a curve is only worth having if it is there before the run is. A tracker that
    was never told a run ended holds a run that was interrupted, which is a fact about the
    experiment rather than an error to hide.

    One tracker records one run, from beginning to end: whoever runs two builds two. Nothing then
    has to carry a run identifier through calls that already know which run they belong to, and a
    run that has ended cannot quietly gain another epoch.
    """

    def begin(self, configuration: ExperimentConfiguration, *, corpus: str, run: str) -> None:
        """Start recording a run of ``configuration`` over the named corpus, called ``run``.

        Raises:
            RuntimeError: If this tracker has already begun a run.
        """
        ...

    def log_epoch(self, outcome: EpochOutcome) -> None:
        """Record what one epoch measured, as soon as it has.

        Raises:
            RuntimeError: If no run has begun, or the run has ended.
        """
        ...

    def end(self, outcome: TrainingOutcome) -> None:
        """Record that the run finished, and what it produced.

        Raises:
            RuntimeError: If no run has begun, or the run has already ended.
        """
        ...
