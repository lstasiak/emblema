from collections.abc import Iterable
from enum import StrEnum

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow


class TaskWindows(StrEnum):
    """Which of a unit's published windows a task reads.

    A corpus is published for pretraining, with as many windows as its units hold; a task may ask
    its question at one moment of a unit only. Which windows it reads is part of what the task
    is, fixed when it is defined, because a choice made per run is a knob that can be turned once
    the numbers are visible.

    Attributes:
        EVERY: Every window of the task's units.
        FIRST: The earliest window of each unit, for a question asked at a fixed moment after the
            unit starts. That the earliest window is the one starting with the unit is the
            ground truth's to check: a window with nothing measured in it is not published.
    """

    EVERY = "every"
    FIRST = "first"

    def chosen_from(self, windows: Iterable[TaskWindow]) -> tuple[TaskWindow, ...]:
        """The windows this choice reads among ``windows``, in the order given."""
        given = tuple(windows)
        if self is TaskWindows.EVERY:
            return given
        earliest: dict[UnitKey, TaskWindow] = {}
        for window in given:
            kept = earliest.get(window.unit)
            if kept is None or window.ends_at < kept.ends_at:
                earliest[window.unit] = window
        chosen = set(earliest.values())
        return tuple(window for window in given if window in chosen)
