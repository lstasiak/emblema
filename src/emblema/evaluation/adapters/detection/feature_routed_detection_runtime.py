from collections.abc import Sequence

from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.ports.detection_runtime import DetectionRuntime


class FeatureRoutedDetectionRuntime:
    """Every kind of detector behind one port, each recipe routed by what it reads a window as.

    A detector over an encoder's states needs the stack that runs the encoder and one over the
    summaries does not, so a process that has both holds two runtimes and routes between them
    here, by the recipe's features rather than by trying one and catching its refusal.
    """

    def __init__(self, *, states: DetectionRuntime, summaries: DetectionRuntime) -> None:
        self._states = states
        self._summaries = summaries

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        runtime = self._states if recipe.features.reads_an_encoder else self._summaries
        return runtime.fit(recipe, task, fitting, scored, retain=retain)
