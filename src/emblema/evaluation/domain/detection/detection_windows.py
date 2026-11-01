from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError, InvalidTaughtEventsError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


@dataclass(frozen=True, kw_only=True)
class DetectionWindows:
    """What one detection run sees: the windows it fits on, answers, and is taught.

    The counterpart of a run's labels for a task that spends no budget of them: the fitting side
    travels as windows alone, so no label reaches the fit but the ones a detector is taught, and
    those travel apart, as the windows of a counted number of the fitting side's events; the
    scored side travels labelled, so the answers are read against what the events say.

    Invariants: at least one window on each side; every taught window is a fitting window and
    overlaps an event.

    Attributes:
        task: The task the windows belong to.
        fitting: Every window of the fitting side the task reads, in block order.
        scored: Every window the run answers, with the event each overlaps.
        taught: The windows of the events the detector is taught; none for a zero-label one.
    """

    task: DownstreamTask
    fitting: tuple[TaskWindow, ...]
    scored: tuple[LabelledWindow, ...]
    taught: tuple[LabelledWindow, ...] = ()

    def __post_init__(self) -> None:
        if not self.fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not self.scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        places = {(str(window.unit), window.position) for window in self.fitting}
        for labelled in self.taught:
            if (str(labelled.window.unit), labelled.window.position) not in places:
                raise InvalidTaughtEventsError(
                    f"the taught window of unit {labelled.window.unit} at "
                    f"{labelled.window.position} is not on the fitting side"
                )
            if not AnomalyScheme.is_event(labelled.target):
                raise InvalidTaughtEventsError(
                    f"a taught window overlaps an event, not {labelled.target}"
                )
