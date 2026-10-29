import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class _Event:
    """One annotated event as the window's question sees it: when, which number, scored or not."""

    number: int
    start: float
    end: float
    scored: bool


class EsaAdRawGroundTruth:
    """Which annotated event, if any, each window of the raw satellite telemetry overlaps.

    The benchmark publishes, beside each mission, ``labels.csv`` with one row per channel an
    event touches, each with the event's identifier and the instants it starts and ends on that
    channel, and ``anomaly_types.csv`` with the category of each event. Only the events touching
    a channel the corpus reads count, since a window holds no other channel; of those, an
    anomaly and a rare nominal event are events alike, as the benchmark scores them, and a gap
    in the downlink is a stretch the task does not score. Events are numbered from one in the
    order the file lists them, among the events that count. A window answers with the lowest
    number of the events it overlaps, zero where it overlaps none, and the excluded mark where
    it overlaps a gap and no event.

    A window ends at the hour the task names and starts a window's length before it, on the
    axis the reader of the corpus placed its observations on: hours since the stated origin.
    Units are named ``<mission>/<YYYY-MM>``, as the reader names them; a mission's files are
    read once and kept.
    """

    _LABELS: ClassVar[str] = "labels.csv"
    _TYPES: ClassVar[str] = "anomaly_types.csv"
    _GAP: ClassVar[str] = "Communication Gap"

    def __init__(
        self,
        root: Path,
        *,
        channels: Mapping[str, Sequence[int]],
        origin_seconds: int,
        window_hours: float,
    ) -> None:
        """Answer for the missions in ``root``, over the channels each mission's corpus reads.

        Args:
            root: The directory holding one directory per mission.
            channels: Channel numbers the corpus reads, per mission; events touching no such
                channel do not count.
            origin_seconds: The epoch second the corpus's hours are counted from.
            window_hours: How long each window is, so its start follows from its end.
        """
        self._root = root
        self._channels = {mission: set(numbers) for mission, numbers in channels.items()}
        self._origin = origin_seconds
        self._window = window_hours
        self._events: dict[str, tuple[_Event, ...]] = {}

    def truths_of(self, corpus: str, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        # This adapter is the truth of one corpus, so the name is the composite's business.
        return {window: self._answer(window) for window in windows}

    def _answer(self, window: TaskWindow) -> float:
        mission, separator, _ = str(window.unit).partition("/")
        if not separator or mission not in self._channels:
            raise UnknownGroundTruthError(f"{window.unit} is not a month of a mission read here")
        end = self._origin + window.ends_at * _SECONDS_PER_HOUR
        start = end - self._window * _SECONDS_PER_HOUR
        overlapping = [
            event for event in self._of_mission(mission) if event.start < end and start < event.end
        ]
        numbered = [event.number for event in overlapping if event.scored]
        if numbered:
            return float(min(numbered))
        if overlapping:
            return AnomalyScheme.EXCLUDED
        return AnomalyScheme.NORMAL

    def _of_mission(self, mission: str) -> tuple[_Event, ...]:
        if mission not in self._events:
            self._events[mission] = self._read(mission)
        return self._events[mission]

    def _read(self, mission: str) -> tuple[_Event, ...]:
        folder = self._root / mission
        categories = self._categories(folder / self._TYPES)
        spans: dict[str, tuple[float, float]] = {}
        order: list[str] = []
        for number, row in enumerate(self._rows(folder / self._LABELS), start=2):
            identifier, channel = row.get("ID"), row.get("Channel")
            if not identifier or not channel or not channel.startswith("channel_"):
                raise UnreadableGroundTruthError(f"{folder / self._LABELS} line {number}: no event")
            try:
                selected = int(channel.removeprefix("channel_")) in self._channels[mission]
                start = self._instant(row["StartTime"])
                end = self._instant(row["EndTime"])
            except (KeyError, ValueError) as error:
                raise UnreadableGroundTruthError(
                    f"{folder / self._LABELS} line {number}: {error}"
                ) from error
            if not selected:
                continue
            if identifier not in spans:
                order.append(identifier)
                spans[identifier] = (start, end)
            else:
                first, last = spans[identifier]
                spans[identifier] = (min(first, start), max(last, end))
        events = []
        counted = 0
        for identifier in order:
            if identifier not in categories:
                raise UnreadableGroundTruthError(
                    f"{folder / self._TYPES} names no category for event {identifier}"
                )
            scored = categories[identifier] != self._GAP
            if scored:
                counted += 1
            start, end = spans[identifier]
            events.append(
                _Event(number=counted if scored else 0, start=start, end=end, scored=scored)
            )
        return tuple(events)

    def _categories(self, path: Path) -> dict[str, str]:
        categories: dict[str, str] = {}
        for number, row in enumerate(self._rows(path), start=2):
            identifier, category = row.get("ID"), row.get("Category")
            if not identifier or category is None:
                raise UnreadableGroundTruthError(f"{path} line {number}: no event or category")
            categories[identifier] = category
        return categories

    @staticmethod
    def _rows(path: Path) -> list[dict[str, str]]:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except OSError as error:
            raise UnreadableGroundTruthError(f"cannot read {path}: {error}") from error

    @staticmethod
    def _instant(text: str) -> float:
        """An instant of the file, as seconds since the epoch."""
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
