from collections.abc import Sequence
from typing import Protocol

from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class ClassicalRuntime(Protocol):
    """Fits a candidate that starts from nothing learnt, and scores it on the windows it is given.

    The counterpart of the runtime that adapts a backbone, and separate from it because the two
    describe a run by different values: one by a transfer mode over named weights, the other by
    a feature scheme over a fit that began empty. A port that took both would have to carry the
    fields of each and let an adapter ignore half of them, and a campaign reading such a record
    could not tell what a candidate was actually set to.

    Nothing of a corpus crosses the port: the task names the manifest its windows sit behind and
    the adapter reads them. Labels of other tasks arrive as sources, already drawn and already
    paired with the task they belong to, so that a fit spanning corpora is a thing the caller
    declared rather than something an adapter decided to do.
    """

    def fit(
        self,
        recipe: ClassicalRecipe,
        task: DownstreamTask,
        sample: LabelSample,
        sources: Sequence[FittingSource],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        """Fit ``recipe`` on ``sample`` and ``sources``, then answer every window of ``scored``.

        The outcome predicts the scored windows in the order given, one each. With ``retain``
        the candidate this fit produced is stored and the outcome names it.

        Raises:
            CandidateNotRetainableError: If ``retain`` was asked of a runtime with nowhere to
                keep what it fits.
            ForeignLabelSampleError: If the sample was drawn from another task.
            InvalidScoredOutcomeError: If there is no window to answer.
            UnreadableTaskCorpusError: If a task's published corpus is not one this can read.
        """
        ...
