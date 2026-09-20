from collections.abc import Mapping, Sequence

from emblema.evaluation.domain.exceptions import UnknownGroundTruthError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow


class InMemoryGroundTruth:
    """Truths given outright, for a task whose ground truth is stated rather than read.

    Stated per unit, as a failure time is, or per window, as an exact reading is; a window is
    answered from its own entry first and from its unit's otherwise.
    """

    def __init__(
        self,
        by_unit: Mapping[UnitKey, float] | None = None,
        by_window: Mapping[TaskWindow, float] | None = None,
    ) -> None:
        self._by_unit = dict(by_unit or {})
        self._by_window = dict(by_window or {})

    def truths_of(self, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        truths: dict[TaskWindow, float] = {}
        missing: list[str] = []
        for window in windows:
            if window in self._by_window:
                truths[window] = self._by_window[window]
            elif window.unit in self._by_unit:
                truths[window] = self._by_unit[window.unit]
            else:
                missing.append(f"{window.unit} at {window.position}")
        if missing:
            raise UnknownGroundTruthError(f"no ground truth known for windows: {sorted(missing)}")
        return truths
