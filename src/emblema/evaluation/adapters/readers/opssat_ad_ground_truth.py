import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow


class OpssatAdGroundTruth:
    """Whether each telemetry fragment is annotated as anomalous, read from the benchmark's file.

    The benchmark marks every fragment whole, so every window of a fragment answers with the
    fragment's own number where it is anomalous and zero where it is not: the fragment is the
    event. Units are named ``<part>/<segment>``, as the reader names them. The file is read once
    and kept.
    """

    _FILE: ClassVar[str] = "segments.csv"
    _SEGMENT: ClassVar[str] = "segment"
    _ANOMALY: ClassVar[str] = "anomaly"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._anomalous: dict[int, bool] | None = None

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        return {window: self._answer(window.unit) for window in windows}

    def _answer(self, unit: UnitKey) -> float:
        _, separator, name = str(unit).partition("/")
        if not separator or not name.isdigit():
            raise UnknownGroundTruthError(f"{unit} is not named <part>/<segment>")
        segment = int(name)
        flags = self._flags()
        if segment not in flags:
            raise UnknownGroundTruthError(f"{unit} is no fragment of the file")
        return float(segment) if flags[segment] else AnomalyScheme.NORMAL

    def _flags(self) -> dict[int, bool]:
        if self._anomalous is None:
            self._anomalous = self._read(self._root / self._FILE)
        return self._anomalous

    def _read(self, path: Path) -> dict[int, bool]:
        flags: dict[int, bool] = {}
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for number, row in enumerate(csv.DictReader(handle), start=2):
                    try:
                        segment, flag = int(row[self._SEGMENT]), row[self._ANOMALY]
                    except (KeyError, ValueError) as error:
                        raise UnreadableGroundTruthError(
                            f"{path} line {number}: {error}"
                        ) from error
                    if flag not in ("0", "1"):
                        raise UnreadableGroundTruthError(
                            f"{path} line {number} marks the fragment as {flag!r}, not 0 or 1"
                        )
                    anomalous = flag == "1"
                    if flags.setdefault(segment, anomalous) != anomalous:
                        raise UnreadableGroundTruthError(
                            f"{path} line {number}: fragment {segment} is marked both ways"
                        )
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path}: {error}") from error
        if not flags:
            raise UnreadableGroundTruthError(f"{path} holds no fragment")
        return flags
