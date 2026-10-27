import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.adapters.readers.text import fields_of, numbered_lines
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

_FILE = "segments.csv"
_SEPARATOR = ","
# Files are hashed in pieces of this size.
_CHUNK_BYTES = 1 << 20


class OpssatAdCorpusReader:
    """Reads the telemetry fragments of OPSSAT-AD, the OPS-SAT anomaly detection benchmark.

    The benchmark publishes ``segments.csv``: one row per reading, each row naming the fragment
    it belongs to, the channel the fragment was cut from, the instant, the value, the nominal
    sampling period, whether the fragment is anomalous and whether the authors put it in their
    training part. A fragment is a unit and reads one channel; the nine channels are the
    satellite's magnetometer and photodiode readings, named as the file names them. Fragments
    differ in length, in sampling period and in the gaps inside them, so the regime promises no
    cadence.

    Units are keyed ``<part>/<segment>``, the part being the authors' ``train`` or ``test``, so a
    publication can hold the authors' test part out by name. Time is seconds since a fragment's
    first row; a row occupies the sampling period that starts at it, so a fragment spans from its
    first row to one period past its last. The labels belong to Evaluation and are not emitted;
    the checksum covers the file whole, labels included, so a relabelled publication is another
    version. The file is read once and kept.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("train", "test")
    # The nine channels, as the file names them: three magnetometer axes and six photodiodes
    # (the benchmark's I_B_FB_MM_0..2 and I_PD1..6_THETA). No physical unit is published.
    CHANNELS: ClassVar[tuple[Channel, ...]] = tuple(
        Channel(name)
        for name in (
            "CADC0872",
            "CADC0873",
            "CADC0874",
            "CADC0884",
            "CADC0886",
            "CADC0888",
            "CADC0890",
            "CADC0892",
            "CADC0894",
        )
    )
    _CHANNEL_NAMES: ClassVar[frozenset[str]] = frozenset(channel.name for channel in CHANNELS)
    _HEADER: ClassVar[tuple[str, ...]] = (
        "channel",
        "timestamp",
        "value",
        "label",
        "sampling",
        "anomaly",
        "segment",
        "train",
    )

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "OPSSAT-AD part")
        self._loaded: dict[UnitKey, _Segment] | None = None

    def describe(self) -> CorpusDescription:
        segments = self._segments()
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(self.CHANNELS)),
            sampling_regime=SamplingRegime.IRREGULAR,
            content=CorpusContent(
                checksum=Checksum.of_chunks(self._chunks_of(self._path())),
                unit_count=len(segments),
                observation_count=sum(len(segment.times) for segment in segments.values()),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for key, segment in self._segments().items():
            yield CorpusUnit(key, TimeExtent(0.0, segment.times[-1] + segment.period))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        part = unit.part
        if part is None or part not in self._subsets:
            raise UnknownUnitError(f"{unit} is not a fragment of a selected part")
        segment = self._segments().get(unit)
        if segment is None:
            raise UnknownUnitError(f"part {part} has no fragment {unit.name}")
        return (
            Observation(segment.channel, time, value)
            for time, value in zip(segment.times, segment.values, strict=True)
        )

    def _path(self) -> Path:
        path = self._root / _FILE
        if not path.is_file():
            raise CorpusDataNotFoundError(f"{path} is missing")
        return path

    @staticmethod
    def _chunks_of(path: Path) -> Iterator[bytes]:
        with path.open("rb") as file:
            while chunk := file.read(_CHUNK_BYTES):
                yield chunk

    def _segments(self) -> dict[UnitKey, "_Segment"]:
        """The selected fragments, parsed once and kept, in the order the file holds them."""
        if self._loaded is None:
            self._loaded = self._parsed(self._path())
        return self._loaded

    def _parsed(self, path: Path) -> dict[UnitKey, "_Segment"]:
        name = path.name
        rows: dict[int, _Rows] = {}
        with path.open("rb") as file:
            lines = numbered_lines(file)
            number, header = next(lines, (0, ""))
            if tuple(header.split(_SEPARATOR)) != self._HEADER:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: expected the columns {self._HEADER}, got {header!r}"
                )
            for number, line in lines:
                self._take(name, number, line, rows)
        kept: dict[UnitKey, _Segment] = {}
        for segment, collected in rows.items():
            part = self.SUBSETS[0] if collected.train else self.SUBSETS[1]
            if part in self._subsets:
                kept[UnitKey.within(part, str(segment))] = collected.sealed()
        return kept

    def _take(self, name: str, number: int, line: str, rows: dict[int, "_Rows"]) -> None:
        """One row into the fragment it belongs to, held to the format and to its fragment."""
        fields = fields_of(name, number, line, _SEPARATOR, len(self._HEADER))
        channel, stamp, value, _, period, anomaly, segment, train = fields
        if channel not in self._CHANNEL_NAMES:
            raise MalformedCorpusDataError(f"{name}, line {number}: unknown channel {channel!r}")
        try:
            instant = datetime.fromisoformat(stamp).timestamp()
            reading = float(value)
            sampling = float(period)
            index = int(segment)
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        if not math.isfinite(reading):
            raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
        if not (math.isfinite(sampling) and sampling > 0):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: sampling period must be positive, got {period!r}"
            )
        if anomaly not in ("0", "1") or train not in ("0", "1"):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: anomaly and train must be 0 or 1, got "
                f"{anomaly!r} and {train!r}"
            )
        if index < 1:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: segment must be a positive number, got {segment!r}"
            )
        collected = rows.get(index)
        if collected is None:
            collected = rows[index] = _Rows(channel, sampling, train == "1", instant)
        collected.add(name, number, channel, sampling, train == "1", instant, reading)


@dataclass
class _Rows:
    """One fragment as it is read: what every row of it has to agree on, and its readings."""

    channel: str
    period: float
    train: bool
    origin: float
    times: list[float] | None = None
    values: list[float] | None = None

    def add(
        self,
        name: str,
        number: int,
        channel: str,
        period: float,
        train: bool,
        instant: float,
        reading: float,
    ) -> None:
        """One more row of the fragment, refused if it disagrees with the fragment's first.

        Raises:
            MalformedCorpusDataError: If the row names another channel, period or part than the
                fragment's first row, or goes back in time.
        """
        if (channel, period, train) != (self.channel, self.period, self.train):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: a fragment's rows share one channel, sampling period "
                "and part"
            )
        if self.times is None or self.values is None:
            self.times, self.values = [], []
        time = instant - self.origin
        if self.times and time < self.times[-1]:
            raise MalformedCorpusDataError(f"{name}, line {number}: goes back in time")
        self.times.append(time)
        self.values.append(reading)

    def sealed(self) -> "_Segment":
        if self.times is None or self.values is None:  # pragma: no cover - a fragment has a row
            raise MalformedCorpusDataError("a fragment without a row")
        return _Segment(self.channel, self.period, tuple(self.times), tuple(self.values))


@dataclass(frozen=True)
class _Segment:
    """One fragment as served: its channel, its nominal period and its readings in order."""

    channel: str
    period: float
    times: tuple[float, ...]
    values: tuple[float, ...]
