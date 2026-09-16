import math
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from emblema.catalog.adapters.readers.subsets import chosen_subsets
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

SUBSETS = ("anomaly-free", "other", "valve1", "valve2")

# The eight sensors in file column order, named as the header names them and united as the
# benchmark's README describes them. The README calls the flow meter `RateRMS`; the files call it
# `Volume Flow RateRMS`, and the files are what is read.
SENSORS = (
    Channel("Accelerometer1RMS", "g"),
    Channel("Accelerometer2RMS", "g"),
    Channel("Current", "A"),
    Channel("Pressure", "bar"),
    Channel("Temperature", "°C"),
    Channel("Thermocouple", "°C"),
    Channel("Voltage", "V"),
    Channel("Volume Flow RateRMS", "L/min"),
)

SEPARATOR = ";"
TIME_COLUMN = "datetime"
# Whether a point is anomalous and whether it opens a collective anomaly. Labels are the language
# of Evaluation, so a version's checksum covers them while its channels do not.
LABEL_COLUMNS = ("anomaly", "changepoint")
HEADER = (TIME_COLUMN, *(sensor.name for sensor in SENSORS))
# One row a second is the testbed's cadence, so the last row occupies the second it starts in and
# the extent ends there. It is a nominal cadence and nothing else rests on it: rows are placed by
# their own timestamps, and the recorded gaps run to several minutes.
SAMPLE_PERIOD = 1.0


class SkabCorpusReader:
    """Reads the experiment files of SKAB, the Skoltech Anomaly Benchmark.

    Each ``data/<folder>/<n>.csv`` holds one experiment on a water-circulation testbed: a
    semicolon-separated row per second with a timestamp, eight sensors and, outside the
    anomaly-free recording, two label columns. An experiment is a unit and a sensor a channel.
    The cadence is nominal rather than fixed — recorded gaps run from two seconds to minutes —
    so observations carry the seconds elapsed since the experiment's first row, not a row index,
    and the regime promises no step.

    Units are keyed ``<folder>/<file>``, since experiments are numbered per folder, and an
    experiment spans from its first row to one second past its last. The checksum covers the
    selected folders' files in folder order and, within a folder, in file-name order, whatever
    order the folders were named in.
    """

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        """Bind the reader to the directory holding the experiment folders.

        Args:
            root: Directory the four experiment folders sit in.
            subsets: Which folders make up the corpus; all four by default. A folder is one
                regime of the testbed, so on its own it is a corpus in its own right.

        Raises:
            ValueError: If no folder is named or a name is not one of the four.
        """
        self._root = root
        self._subsets = chosen_subsets(subsets, SUBSETS, "SKAB folder")

    def describe(self) -> CorpusDescription:
        """Validate the selected experiments and describe them.

        Raises:
            CorpusDataNotFoundError: If a selected folder is missing or holds no experiment.
            MalformedCorpusDataError: If a header is not the documented one, a row has the wrong
                width, a value is not a finite number, a timestamp is unreadable or a row goes
                back in time.
        """
        rows: list[int] = []

        def contents() -> Iterator[bytes]:
            """The files in canonical order, each counted as it passes, then let go."""
            for path in self._paths():
                content = path.read_bytes()
                rows.append(self._span_of(path.name, content)[0])
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(SENSORS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(rows),
                observation_count=sum(rows) * len(SENSORS),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        """Every experiment of the selected folders, in canonical order.

        Raises:
            CorpusDataNotFoundError: If a selected folder is missing or holds no experiment.
            MalformedCorpusDataError: As for ``describe``.
        """
        for path in self._paths():
            _, span = self._span_of(path.name, path.read_bytes())
            yield CorpusUnit(self._key_of(path), TimeExtent(0.0, span))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """The eight sensor values of every row of one experiment, row by row.

        Raises:
            UnknownUnitError: If the key does not name an experiment of a selected folder.
            CorpusDataNotFoundError: If the experiment's folder is missing.
            MalformedCorpusDataError: If a row of the experiment is malformed or goes back in
                time.
        """
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
        return UnitKey(f"{path.parent.name}/{path.stem}")

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
        return rows, last + SAMPLE_PERIOD

    @classmethod
    def _observations_of(cls, name: str, content: bytes) -> Iterator[Observation]:
        for time, values in cls._rows_of(name, content):
            for sensor, value in zip(SENSORS, values, strict=True):
                yield Observation(sensor.name, time, value)

    @classmethod
    def _rows_of(cls, name: str, content: bytes) -> Iterator[tuple[float, list[float]]]:
        """Every data row as seconds since the first row and the eight sensor values."""
        # Lines break on a newline and nothing else; the carriage return of the files written on
        # Windows belongs to the line ending, not to the last field.
        lines = content.decode("utf-8", errors="replace").split("\n")
        header: tuple[str, ...] | None = None
        start: datetime | None = None
        last = -math.inf
        for number, line in enumerate(lines, 1):
            fields = line.rstrip("\r").split(SEPARATOR)
            # A blank line carries nothing: it is not the end of the experiment, and taking it
            # for one would truncate the file without a word.
            if fields == [""]:
                continue
            if header is None:
                header = cls._validated_header(name, number, tuple(fields))
                continue
            moment, values = cls._parse_row(name, number, header, fields)
            if start is None:
                start = moment
            time = (moment - start).total_seconds()
            if time < last:
                raise MalformedCorpusDataError(f"{name}, line {number}: goes back in time")
            last = time
            yield time, values
        if header is None:
            raise MalformedCorpusDataError(f"{name}: no header")

    @staticmethod
    def _validated_header(name: str, number: int, fields: tuple[str, ...]) -> tuple[str, ...]:
        if fields[: len(HEADER)] != HEADER:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected the columns {HEADER}, got {fields}"
            )
        trailing = fields[len(HEADER) :]
        # The anomaly-free recording has no labels to carry, so both label columns are absent
        # together; anything else is a file this reader does not know how to read.
        if trailing not in ((), LABEL_COLUMNS):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected no columns after {HEADER[-1]!r} or "
                f"{LABEL_COLUMNS}, got {trailing}"
            )
        return fields

    @staticmethod
    def _parse_row(
        name: str, number: int, header: tuple[str, ...], fields: list[str]
    ) -> tuple[datetime, list[float]]:
        if len(fields) != len(header):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected {len(header)} columns, got {len(fields)}"
            )
        try:
            moment = datetime.fromisoformat(fields[0])
            values = [float(field) for field in fields[1 : len(HEADER)]]
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        if not all(math.isfinite(value) for value in values):
            raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
        return moment, values
