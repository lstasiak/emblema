from collections.abc import Mapping, Sequence
from typing import Protocol

from emblema.evaluation.domain.labels.task_window import TaskWindow


class GroundTruth(Protocol):
    """Tells, for each window of a task, the one fact its label is read from.

    The Catalog publishes measurements, never answers: a corpus that carried its labels would
    hand the pretraining an axis to learn the task from without anyone asking for it. So what a
    supervised task needs beyond the windows enters here, from wherever the ground truth of that
    corpus is published. What the number means is the task's label scheme's business — the
    moment a unit failed for a remaining-life task, the exact reading a window is asked to
    forecast for a forecasting one — and the scheme turns it into the target. Windows in and one
    number per window out, so a truth that varies along a unit and one that does not travel the
    same way.
    """

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        """What the ground truth says about each of those windows of that corpus.

        The corpus is named on every call rather than fixed when the adapter is built, because a
        candidate fitted across corpora draws labels from several of them in one run, and a unit
        key is only unique within the corpus it was cut from.

        Which corpora a process can answer for is a fact about the process, so it is a register
        of readers that refuses a corpus nobody put in it; a reader of one corpus, reached only
        through such a register, answers whatever name it is passed.

        Raises:
            UnknownGroundTruthError: If the ground truth says nothing about one of the windows,
                or — asked of a register — nothing at all about that corpus.
        """
        ...
