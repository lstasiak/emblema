from collections.abc import Sequence

from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.exceptions import UnsupportedDetectorFeaturesError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.ports.detection_runtime import DetectionRuntime


class FeatureRoutedDetectionRuntime:
    """Every kind of detector behind one port, each recipe routed by what it reads a window as.

    A detector over an encoder's states, or over the cells the encoder would read, needs the
    stack that lays them out and runs the encoder and one over the summaries does not, and one
    scored by the error of a backbone's pretext needs the backbone's objective whole, so a process
    holds a runtime for each it can run and routes between them here, by the recipe's features
    rather than by trying one and catching its refusal.
    """

    def __init__(
        self,
        *,
        states: DetectionRuntime,
        summaries: DetectionRuntime,
        pretext: DetectionRuntime | None = None,
    ) -> None:
        self._states = states
        self._summaries = summaries
        self._pretext = pretext

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        if recipe.features.asks_the_pretext:
            if self._pretext is None:
                raise UnsupportedDetectorFeaturesError(
                    f"this process asks no backbone's pretext, so it cannot fit {recipe.features}"
                )
            return self._pretext.fit(recipe, task, fitting, scored, retain=retain)
        runtime = self._states if recipe.features.lays_out_cells else self._summaries
        return runtime.fit(recipe, task, fitting, scored, retain=retain)
