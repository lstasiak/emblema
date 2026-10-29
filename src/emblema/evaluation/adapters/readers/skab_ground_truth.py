from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow


class SkabGroundTruth:
    """Which experiment's anomaly, if any, each window of the testbed's recordings overlaps.

    Every experiment file of the benchmark marks each of its rows as anomalous or not, and
    every experiment holds one anomaly, so the events are the experiments: a window overlapping
    a row marked anomalous answers with its experiment's number, counted from one over the
    experiments in the order their keys sort, and zero otherwise. The anomaly-free recording
    carries no marks and every window of it is normal.

    A window ends at the second the task names and starts a window's length before it, on the
    axis the reader of the corpus placed its observations on: seconds since the experiment's
    first row. Units are named ``<folder>/<file>``, as the reader names them; a file is read
    once and kept.
    """

    _SEPARATOR: ClassVar[str] = ";"
    _TIME: ClassVar[str] = "datetime"
    _ANOMALY: ClassVar[str] = "anomaly"

    def __init__(self, root: Path, *, window_seconds: float) -> None:
        self._root = root
        self._window = window_seconds
        self._marked: dict[UnitKey, tuple[float, ...]] = {}
        self._numbers: dict[UnitKey, int] | None = None

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        return {window: self._answer(window) for window in windows}

    def _answer(self, window: TaskWindow) -> float:
        start = window.ends_at - self._window
        marked = self._marked_seconds(window.unit)
        if any(start <= second < window.ends_at for second in marked):
            return float(self._number_of(window.unit))
        return AnomalyScheme.NORMAL

    def _number_of(self, unit: UnitKey) -> int:
        if self._numbers is None:
            experiments = sorted(
                (
                    UnitKey(f"{path.parent.name}/{path.stem}")
                    for path in self._root.glob("*/*.csv")
                    if path.parent.name != "anomaly-free"
                ),
                key=str,
            )
            self._numbers = {key: number for number, key in enumerate(experiments, start=1)}
        return self._numbers[unit]

    def _marked_seconds(self, unit: UnitKey) -> tuple[float, ...]:
        if unit not in self._marked:
            self._marked[unit] = self._read(unit)
        return self._marked[unit]

    def _read(self, unit: UnitKey) -> tuple[float, ...]:
        """The seconds since the first row of every row the file marks as anomalous."""
        folder, separator, name = str(unit).partition("/")
        if not separator or not name:
            raise UnknownGroundTruthError(f"{unit} is not an experiment of a folder")
        path = self._root / folder / f"{name}.csv"
        if path.parent != self._root / folder or not path.is_file():
            raise UnknownGroundTruthError(f"{unit} is no experiment under {self._root}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path}: {error}") from error
        if not lines:
            raise UnreadableGroundTruthError(f"{path} is empty")
        header = lines[0].split(self._SEPARATOR)
        if self._TIME not in header:
            raise UnreadableGroundTruthError(f"{path} has no column {self._TIME}")
        if self._ANOMALY not in header:
            return ()
        time_at, anomaly_at = header.index(self._TIME), header.index(self._ANOMALY)
        marked = []
        first: datetime | None = None
        for number, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            fields = line.split(self._SEPARATOR)
            if len(fields) != len(header):
                raise UnreadableGroundTruthError(
                    f"{path} line {number} has {len(fields)} fields, not {len(header)}"
                )
            try:
                moment = datetime.fromisoformat(fields[time_at])
                flag = float(fields[anomaly_at])
            except ValueError as error:
                raise UnreadableGroundTruthError(f"{path} line {number}: {error}") from error
            if first is None:
                first = moment
            if flag not in (0.0, 1.0):
                raise UnreadableGroundTruthError(
                    f"{path} line {number} marks the row as {fields[anomaly_at]!r}, not 0 or 1"
                )
            if flag:
                marked.append((moment - first).total_seconds())
        return tuple(marked)
