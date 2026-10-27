from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


@dataclass(frozen=True, kw_only=True)
class DetectionWindows:
    """What one detection run sees: the windows it fits on, unlabelled, and the ones it answers.

    The counterpart of a run's labels for a task that spends none: the fitting side travels as
    windows alone, so no label can reach the fit by accident, and the scored side travels
    labelled, so the answers are read against what the events say.

    Invariants: at least one window on each side.

    Attributes:
        task: The task the windows belong to.
        fitting: Every window of the fitting side the task reads, in block order.
        scored: Every window the run answers, with the event each overlaps.
    """

    task: DownstreamTask
    fitting: tuple[TaskWindow, ...]
    scored: tuple[LabelledWindow, ...]

    def __post_init__(self) -> None:
        if not self.fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not self.scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
