"""Fitting a detector: the one step of a detection cell that is the runtime's own."""

from emblema.evaluation.adapters.in_memory.detection_runtime import InMemoryDetectionRuntime
from emblema.evaluation.application.use_cases.run_detection import (
    RunDetection,
    RunDetectionCommand,
)
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from tests.evaluation.application.use_cases.test_draw_detection_windows import drawn
from tests.evaluation.support import TASK

RECIPE = DetectorRecipe(
    features=DetectorFeatures.SUMMARIES, reading=None, neighbours=3, contamination=0.05, seed=1
)


def test_the_runtime_is_handed_the_fitting_windows_and_answers_the_scored_ones() -> None:
    runtime = InMemoryDetectionRuntime()
    windows = drawn()
    use_case = RunDetection(lambda command: windows, runtime)

    outcome = use_case(RunDetectionCommand(task=TASK, recipe=RECIPE, purpose=RunPurpose.TUNING))

    assert runtime.detections[0].recipe == RECIPE
    assert runtime.detections[0].fitting == windows.fitting
    assert [p.window for p in outcome.predictions] == [w.window for w in windows.scored]
    # The fake sets its alarm at the latest end the fit saw, 180: the window ending there scores
    # zero and nothing beyond it, so no alarm is raised on the validation side.
    assert [p.predicted for p in outcome.predictions] == [-120.0, -60.0, 0.0]
