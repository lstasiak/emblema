from collections.abc import Callable
from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_detection_windows import (
    DrawDetectionWindowsCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.detection.detection_windows import DetectionWindows
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.detection_runtime import DetectionRuntime


@dataclass(frozen=True, kw_only=True)
class RunDetectionCommand:
    """One cell of a detection campaign: a recipe over a task, for a purpose.

    Attributes:
        task: Task to fit on and to be scored on.
        recipe: How the detector is made; it carries the seed of the fit.
        purpose: What the run is for, which decides which side it is scored on.
        retain: Whether the detector this run fits is kept as an artifact.
    """

    task: TaskId
    recipe: DetectorRecipe
    purpose: RunPurpose = RunPurpose.TUNING
    retain: bool = False


class RunDetection:
    """Fits a detector on a task's fitting side, taught what its recipe asks, and scores it.

    Which windows those are is not this use case's business: reading them, and drawing the few
    events a recipe is taught, is the same for every detector a campaign compares, so it is done
    in one place and this one is left with the single step that is its own.
    """

    def __init__(
        self,
        draw_detection_windows: Callable[[DrawDetectionWindowsCommand], DetectionWindows],
        runtime: DetectionRuntime,
    ) -> None:
        self._windows = draw_detection_windows
        self._runtime = runtime

    def __call__(self, command: RunDetectionCommand) -> ScoredOutcome:
        """Read, fit, score; return the detector's score for every scored window.

        Raises:
            TaskNotFoundError: If the task is unknown.
            ProtocolMismatchError: If the task spends labels, or the run is a selection.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a scored window, or
                about a fitting window where events are to be taught.
            InvalidTaughtEventsError: If the fitting side holds fewer events than the recipe is
                taught.
            UnsupportedDetectorFeaturesError: If the runtime cannot produce the recipe's
                features.
            CandidateNotRetainableError: If the run was to keep what it fitted and the runtime
                has nowhere to keep it.
        """
        windows = self._windows(
            DrawDetectionWindowsCommand(
                task=command.task,
                purpose=command.purpose,
                events=command.recipe.events,
                seed=command.recipe.seed,
            )
        )
        return self._runtime.fit(
            command.recipe,
            windows.task,
            windows.fitting,
            windows.scored,
            taught=windows.taught,
            retain=command.retain,
        )
