from collections.abc import Mapping, Sequence
from pathlib import Path

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow


class CmapssGroundTruth:
    """When each turbofan failed, counted from the run-to-failure files the corpus was read from.

    The ground truth of this corpus is not published beside it: each training engine is recorded
    until it fails, so the failure is the end of its record and the answer is the length of the
    record. The moment is expressed the way the corpus times its units — the first cycle at one,
    the axis running to one past the last — so that a window's end and a failure are comparable
    without either side knowing how the other was built. Every window of an engine is answered
    with the same moment: the truth of a remaining-life task is a fact about the unit.

    Units are named ``<subset>/<engine>``, as the reader of this corpus names them.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lengths: dict[str, dict[int, int]] = {}

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        failures = self._failure_times({window.unit for window in windows})
        return {window: failures[window.unit] for window in windows}

    def _failure_times(self, units: set[UnitKey]) -> dict[UnitKey, float]:
        failures = {}
        for unit in sorted(units, key=str):
            subset, engine = self._parsed(unit)
            lengths = self._of_subset(subset, unit)
            if engine not in lengths:
                raise UnknownGroundTruthError(f"{unit} is not an engine of subset {subset}")
            # The axis starts at cycle one and the extent runs one past the last cycle, so an
            # engine recorded for n cycles fails at n + 1.
            failures[unit] = float(lengths[engine] + 1)
        return failures

    @staticmethod
    def _parsed(unit: UnitKey) -> tuple[str, int]:
        subset, _, engine = str(unit).partition("/")
        if not subset or not engine.isdigit():
            raise UnknownGroundTruthError(f"{unit} is not named <subset>/<engine>")
        return subset, int(engine)

    def _of_subset(self, subset: str, unit: UnitKey) -> dict[int, int]:
        if subset not in self._lengths:
            self._lengths[subset] = self._counted(self._root / f"train_{subset}.txt", unit)
        return self._lengths[subset]

    @staticmethod
    def _counted(path: Path, unit: UnitKey) -> dict[int, int]:
        lengths: dict[int, int] = {}
        try:
            content = path.read_text()
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path} for {unit}: {error}") from error
        for number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            engine = line.split()[0]
            if not engine.isdigit():
                raise UnreadableGroundTruthError(
                    f"{path} line {number} does not start with an engine number: {engine!r}"
                )
            lengths[int(engine)] = lengths.get(int(engine), 0) + 1
        if not lengths:
            raise UnreadableGroundTruthError(f"{path} holds no engine")
        return lengths
