from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class DetectionRuntime(Protocol):
    """Fits a detector on the fitting side's windows, and scores the windows it is given.

    Apart from the runtimes that adapt a backbone or fit a classical candidate because both
    take a sample of labels, and a detector takes no budget of them: a port that handed it one
    would hand it an axis the task does not have. What a detector may be taught is the windows
    of a counted number of the fitting side's events, handed apart from the side itself. Nothing
    of a corpus crosses the port: the task names the manifest its windows sit behind and the
    adapter reads them.
    """

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        taught: Sequence[LabelledWindow] = (),
        retain: bool,
    ) -> ScoredOutcome:
        """Fit ``recipe`` on ``fitting``, taught ``taught``, and answer every window of ``scored``.

        The outcome scores the windows in the order given, one each, as the score above the
        alarm the recipe sets. With ``retain`` the detector this fit produced is stored and the
        outcome names it.

        Raises:
            UnsupportedDetectorFeaturesError: If the recipe reads features this runtime cannot
                produce.
            InvalidTaughtEventsError: If the recipe is taught events and the taught windows do
                not hold that many, or lie off the fitting side, or a recipe taught none is
                handed some.
            CandidateNotRetainableError: If ``retain`` was asked of a runtime with nowhere to
                keep what it fits.
            InvalidScoredOutcomeError: If there is no window to fit on or to answer.
            UnreadableTaskCorpusError: If the task's published corpus is not one this can read.
            UnknownBackboneError: If the recipe names pretrained weights the runtime cannot
                supply.
        """
        ...
