import math
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.adapters.readers.text import fields_of, finite_floats, numbered_lines
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

_SEPARATOR = ";"
_TIME_COLUMN = "datetime"
# Whether a point is anomalous and whether it opens a collective anomaly. Labels are the language
# of Evaluation, so a version's checksum covers them while its channels do not.
_LABEL_COLUMNS = ("anomaly", "changepoint")
# The last row occupies the second it starts in, so an extent ends one cadence past it. This is
# the only thing the cadence is used for.
_SAMPLE_PERIOD = 1.0


class SkabCorpusReader:
    """Reads the experiment files of SKAB, the Skoltech Anomaly Benchmark.

    Each ``data/<folder>/<n>.csv`` holds one experiment on a water-circulation testbed: a
    semicolon-separated row per second with a timestamp, eight sensors and, outside the
    anomaly-free recording, two label columns. An experiment is a unit and a sensor a channel.
    The cadence is nominal rather than fixed — recorded gaps run from two seconds to minutes —
    so observations carry the seconds elapsed since the experiment's first row, not a row index,
    and the regime promises no step.

    The reader is bound to a selection of the four folders, each one regime of the testbed and a
    corpus in its own right. Units are keyed ``<folder>/<file>``, since experiments are numbered
    per folder, and an experiment spans from its first row to one second past its last. The
    checksum covers the selected folders' files in folder order and, within a folder, in file-name
    order, whatever order the folders were named in.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("anomaly-free", "other", "valve1", "valve2")
    # The eight sensors in file column order, named as the header names them and united as the
    # benchmark's README describes them. The README calls the flow meter `RateRMS`; the files call
    # it `Volume Flow RateRMS`, and the files are what is read.
    SENSORS: ClassVar[tuple[Channel, ...]] = (
        Channel("Accelerometer1RMS", "g"),
        Channel("Accelerometer2RMS", "g"),
        Channel("Current", "A"),
        Channel("Pressure", "bar"),
        Channel("Temperature", "°C"),
        Channel("Thermocouple", "°C"),
        Channel("Voltage", "V"),
        Channel("Volume Flow RateRMS", "L/min"),
    )
    _HEADER: ClassVar[tuple[str, ...]] = (_TIME_COLUMN, *(sensor.name for sensor in SENSORS))

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "SKAB folder")

    def describe(self) -> CorpusDescription:
        rows: list[int] = []

        def contents() -> Iterator[bytes]:
            for path in self._paths():
                content = path.read_bytes()
                rows.append(self._span_of(path.name, content)[0])
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(self.SENSORS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(rows),
                observation_count=sum(rows) * len(self.SENSORS),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for path in self._paths():
            _, span = self._span_of(path.name, path.read_bytes())
            yield CorpusUnit(self._key_of(path), TimeExtent(0.0, span))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        path = self._locate(unit)
        return self._observations_of(path.name, path.read_bytes())

    def _paths(self) -> Iterator[Path]:
        """The selected files in canonical order: folders as declared, files by name within one."""
        for subset in self._subsets:
            folder = self._root / subset
            if not folder.is_dir():
                raise CorpusDataNotFoundError(f"{folder} is missing")
            files = sorted(folder.glob("*.csv"), key=lambda path: path.name)
            if not files:
                raise CorpusDataNotFoundError(f"{folder} holds no experiment")
            yield from files

    def _key_of(self, path: Path) -> UnitKey:
        return UnitKey.within(path.parent.name, path.stem)

    def _locate(self, unit: UnitKey) -> Path:
        subset, separator, experiment = unit.value.partition("/")
        if not separator or subset not in self._subsets:
            raise UnknownUnitError(f"{unit} is not an experiment of a selected folder")
        folder = self._root / subset
        if not folder.is_dir():
            raise CorpusDataNotFoundError(f"{folder} is missing")
        path = folder / f"{experiment}.csv"
        # A key naming anything but a file of the folder — a nested path, a step upwards — names
        # no experiment, however the file system would resolve it.
        if path.parent != folder or not path.is_file():
            raise UnknownUnitError(f"{folder} has no experiment {experiment}")
        return path

    @classmethod
    def _span_of(cls, name: str, content: bytes) -> tuple[int, float]:
        """How many rows the experiment holds and how far its time extent reaches."""
        rows = 0
        last = 0.0
        for time, _ in cls._rows_of(name, content):
            rows += 1
            last = time
        if not rows:
            raise MalformedCorpusDataError(f"{name}: no row")
        return rows, last + _SAMPLE_PERIOD

    @classmethod
    def _observations_of(cls, name: str, content: bytes) -> Iterator[Observation]:
        for time, values in cls._rows_of(name, content):
            for sensor, value in zip(cls.SENSORS, values, strict=True):
                yield Observation(sensor.name, time, value)

    @classmethod
    def _rows_of(cls, name: str, content: bytes) -> Iterator[tuple[float, list[float]]]:
        """Every data row as seconds since the first row and the eight sensor values."""
        header: tuple[str, ...] | None = None
        start: datetime | None = None
        last = -math.inf
        for number, line in numbered_lines(content.split(b"\n")):
            if header is None:
                header = cls._validated_header(name, number, tuple(line.split(_SEPARATOR)))
                continue
            moment, values = cls._parse_row(name, number, header, line)
            if start is None:
                start = moment
            time = (moment - start).total_seconds()
            if time < last:
                raise MalformedCorpusDataError(f"{name}, line {number}: goes back in time")
            last = time
            yield time, values
        if header is None:
            raise MalformedCorpusDataError(f"{name}: no header")

    @classmethod
    def _validated_header(cls, name: str, number: int, fields: tuple[str, ...]) -> tuple[str, ...]:
        if fields[: len(cls._HEADER)] != cls._HEADER:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected the columns {cls._HEADER}, got {fields}"
            )
        trailing = fields[len(cls._HEADER) :]
        # The anomaly-free recording has no labels to carry, so both label columns are absent
        # together; anything else is a file this reader does not know how to read.
        if trailing not in ((), _LABEL_COLUMNS):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected no columns after {cls._HEADER[-1]!r} or "
                f"{_LABEL_COLUMNS}, got {trailing}"
            )
        return fields

    @classmethod
    def _parse_row(
        cls, name: str, number: int, header: tuple[str, ...], line: str
    ) -> tuple[datetime, list[float]]:
        fields = fields_of(name, number, line, _SEPARATOR, len(header))
        try:
            moment = datetime.fromisoformat(fields[0])
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        return moment, finite_floats(name, number, fields[1 : len(cls._HEADER)])
