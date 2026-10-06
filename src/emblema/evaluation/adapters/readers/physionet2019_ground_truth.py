from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnlabelledWindowError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow


@dataclass(frozen=True)
class _Course:
    """What a stay's file says about the task: where it starts and when its label first turns."""

    first_hour: int
    labelled_from: int | None


class Physionet2019GroundTruth:
    """Whether sepsis follows a stay's first window, read from the challenge's files.

    Every row of a stay carries the challenge's label, a bit that turns to one six hours before
    the onset the challenge dates and stays there. The task asks at the end of a stay's first
    window, as recorded from its first row, whether that label turns later in the stay; the answer
    is one if it does and zero if it never does.

    Two windows have no answer and are refused rather than labelled. A window that does not end
    one window length after the stay's first hour is not the moment the question is asked at: a
    task reading every window, or a stay whose first day held no measurement and whose earliest
    published window starts later, would otherwise be answered as if it were. And a window in
    which the label has already turned shows the treatment its own label follows from.

    Units are named ``training_set<X>/p<nnnnnn>``, as the reader of this corpus names them, and
    the label is the column the scheme names. A stay's file is read once and kept.
    """

    _SEPARATOR: ClassVar[str] = "|"
    _HOUR: ClassVar[str] = "ICULOS"
    _SET_PREFIX: ClassVar[str] = "training_set"
    _EXTENSION: ClassVar[str] = ".psv"

    def __init__(self, root: Path, scheme: OutcomeScheme, window: float) -> None:
        self._root = root
        self._label = scheme.outcome
        self._window = window
        self._courses: dict[UnitKey, _Course] = {}

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        return {window: self._answer(window) for window in windows}

    def _answer(self, window: TaskWindow) -> float:
        course = self._course_of(window.unit)
        if window.ends_at != course.first_hour + self._window:
            raise UnlabelledWindowError(
                f"{window.unit}: the window ending at {window.ends_at:g} is not the stay's first "
                f"{self._window:g} hours from hour {course.first_hour}"
            )
        if course.labelled_from is None:
            return 0.0
        if course.labelled_from < window.ends_at:
            raise UnlabelledWindowError(
                f"{window.unit}: {self._label} turns at hour {course.labelled_from}, inside the "
                f"window ending at {window.ends_at:g}"
            )
        return 1.0

    def _course_of(self, unit: UnitKey) -> _Course:
        if unit not in self._courses:
            self._courses[unit] = self._read(self._path_of(unit))
        return self._courses[unit]

    def _path_of(self, unit: UnitKey) -> Path:
        subset, _, stay = str(unit).partition("/")
        if not subset.startswith(self._SET_PREFIX) or not stay or "/" in stay:
            raise UnknownGroundTruthError(f"{unit} is not named training_set<X>/<stay>")
        path = self._root / subset / f"{stay}{self._EXTENSION}"
        # A set that is there without the stay names a stay the challenge never recorded; a set
        # that is not there is a corpus this process was given without its files.
        if path.parent.is_dir() and not path.is_file():
            raise UnknownGroundTruthError(f"{unit} is no stay of {subset}")
        return path

    def _read(self, path: Path) -> _Course:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path}: {error}") from error
        if not lines:
            raise UnreadableGroundTruthError(f"{path} is empty")
        header = lines[0].split(self._SEPARATOR)
        for needed in (self._HOUR, self._label):
            if needed not in header:
                raise UnreadableGroundTruthError(f"{path} has no column {needed}")
        hour_at, label_at = header.index(self._HOUR), header.index(self._label)
        first_hour: int | None = None
        labelled_from: int | None = None
        for number, line in enumerate(lines[1:], start=2):
            fields = line.split(self._SEPARATOR)
            if len(fields) != len(header):
                raise UnreadableGroundTruthError(
                    f"{path} line {number} has {len(fields)} fields, not {len(header)}"
                )
            hour, label = fields[hour_at], fields[label_at]
            if not (hour.isascii() and hour.isdigit()) or label not in ("0", "1"):
                raise UnreadableGroundTruthError(
                    f"{path} line {number} records hour {hour!r} and {self._label} {label!r}"
                )
            first_hour = int(hour) if first_hour is None else first_hour
            if label == "1" and labelled_from is None:
                labelled_from = int(hour)
        if first_hour is None:
            raise UnreadableGroundTruthError(f"{path} has no row")
        return _Course(first_hour, labelled_from)
