from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow


class Physionet2012GroundTruth:
    """What each intensive-care stay recorded for one outcome, read from the challenge's files.

    The challenge publishes the outcomes of every stay of a set in one file beside it —
    ``Outcomes-a.txt`` for ``set-a`` — a row per stay keyed by its record number, one column per
    outcome. Which column is the truth is the scheme's outcome, named once with the task, so the
    task and the answers it is given cannot drift apart. Every window of a stay is answered with
    the stay's outcome: the truth of this task is a fact about the stay, not about a moment in it.

    Units are named ``set-<x>/<RecordID>``, as the reader of this corpus names them. A set's file
    is read once and kept.
    """

    _RECORD: ClassVar[str] = "RecordID"
    _SET_PREFIX: ClassVar[str] = "set-"

    def __init__(self, root: Path, scheme: OutcomeScheme) -> None:
        self._root = root
        self._outcome = scheme.outcome
        self._sets: dict[str, dict[str, float]] = {}

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        recorded = {unit: self._recorded(unit) for unit in {window.unit for window in windows}}
        return {window: recorded[window.unit] for window in windows}

    def _recorded(self, unit: UnitKey) -> float:
        subset, _, record = str(unit).partition("/")
        if not subset.startswith(self._SET_PREFIX) or not record:
            raise UnknownGroundTruthError(f"{unit} is not named set-<x>/<RecordID>")
        outcomes = self._of_set(subset)
        if record not in outcomes:
            raise UnknownGroundTruthError(f"{unit} has no outcome recorded in its set")
        return outcomes[record]

    def _of_set(self, subset: str) -> dict[str, float]:
        if subset not in self._sets:
            letter = subset.removeprefix(self._SET_PREFIX)
            self._sets[subset] = self._read(self._root / f"Outcomes-{letter}.txt")
        return self._sets[subset]

    def _read(self, path: Path) -> dict[str, float]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path}: {error}") from error
        if not lines:
            raise UnreadableGroundTruthError(f"{path} is empty")
        header = lines[0].split(",")
        for needed in (self._RECORD, self._outcome):
            if needed not in header:
                raise UnreadableGroundTruthError(f"{path} has no column {needed}")
        record_at, outcome_at = header.index(self._RECORD), header.index(self._outcome)
        outcomes: dict[str, float] = {}
        for number, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            fields = line.split(",")
            if len(fields) != len(header):
                raise UnreadableGroundTruthError(
                    f"{path} line {number} has {len(fields)} fields, not {len(header)}"
                )
            record = fields[record_at]
            if not record.isdigit() or record in outcomes:
                raise UnreadableGroundTruthError(
                    f"{path} line {number} names record {record!r}, which is not a new number"
                )
            try:
                outcomes[record] = float(fields[outcome_at])
            except ValueError as error:
                raise UnreadableGroundTruthError(
                    f"{path} line {number} records {fields[outcome_at]!r} for {self._outcome}"
                ) from error
        return outcomes
