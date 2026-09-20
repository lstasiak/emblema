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

    def truths_of(self, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        """What the ground truth says about each of those windows.

        Raises:
            UnknownGroundTruthError: If the ground truth says nothing about one of the windows.
        """
        ...
