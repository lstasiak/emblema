from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class PatchRuntime(Protocol):
    """Trains a patch model from nothing on a task's labels, and scores it on the windows given.

    Neither of the other two runtimes, because it is described by neither's values. It adapts
    no backbone, so a transfer mode and named weights would be fields it ignores; and it is a
    network held to the compute budget the adapted arms share, trained on the stack they are,
    so the runtime that fits classical methods by their own procedure is the wrong home too.

    Nothing of a corpus crosses the port: the task names the manifest its windows sit behind,
    and laying them on a grid is the adapter's business. Weights never cross it either: a run
    asked to keep what it trained answers with a reference to it.
    """

    def train(
        self,
        plan: PatchPlan,
        task: DownstreamTask,
        sample: LabelSample,
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        """Learn the task from ``sample`` under ``plan`` and answer every window of ``scored``.

        The outcome predicts the scored windows in the order given, one each. With ``retain``
        the model this run trained is stored and the outcome names it.

        Raises:
            CandidateNotRetainableError: If ``retain`` was asked of a runtime with nowhere to
                keep what it trains.
            ForeignLabelSampleError: If the sample was drawn from another task.
            InvalidScoredOutcomeError: If there is no window to answer.
            InvalidPatchModelSpecError: If a window of the task's corpus is laid on fewer
                steps than one patch covers.
            DivergedAdaptationError: If the training loss stops being finite.
        """
        ...
