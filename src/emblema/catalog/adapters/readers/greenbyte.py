import csv
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.adapters.readers.excluded_months import ExcludedMonths
from emblema.catalog.adapters.readers.file_chunks import chunks_of
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

# The origin of every farm's time axis: midnight UTC of the first day of the year both farms'
# tables begin in, stated rather than read off the data so that two farms lie on one axis.
ORIGIN_SECONDS = 1_451_606_400
ORIGIN_TEXT = "2016-01-01T00:00:00Z"
_SECONDS_PER_HOUR = 3600
_TIME_COLUMN = "# Date and time"
_TIME_ZONE = "# Time zone: UTC"
# The lines a table's preamble may run to before its header; the exports seen hold nine.
_PREAMBLE_LIMIT = 64
_TABLE = re.compile(r"Turbine_Data_(?P<farm>[^_]+)_(?P<turbine>[^_]+)_\d{4}-\d{2}-\d{2}_-_.*\.csv")
# How the exports have spelt a unit beside the spelling a channel states; any other unit under a
# signal's title is another quantity, and is refused.
_UNIT_SPELLINGS = {"mm/s2": ("mm/s2", "mm/ss")}


@dataclass(frozen=True)
class Signal:
    """One quantity a turbine's controller sends, as the export titles it.

    Attributes:
        name: The title, which is the channel's name.
        unit: The unit the export states for it; ``None`` where it states none.
    """

    name: str
    unit: str | None

    def holds(self, column: str) -> bool:
        """Whether a header column holds this signal's averages.

        It does where it is the signal's title, whatever its case, followed by the signal's unit
        in brackets where it has one; a title followed by a comma is a statistic of the signal
        (its minimum, its deviation) and is not.

        Raises:
            ValueError: If the column carries the title with another unit, or with none where
                the signal has one: the same name over another quantity.
        """
        stated: str | None = None
        if column.casefold() != self.name.casefold():
            title, bracket, rest = column.rpartition(" (")
            if not bracket or not rest.endswith(")") or title.casefold() != self.name.casefold():
                return False
            stated = rest[:-1]
        spellings = (None,) if self.unit is None else _UNIT_SPELLINGS.get(self.unit, (self.unit,))
        if stated not in spellings:
            raise ValueError(
                f"{column!r} states {stated or 'no unit'} where {self.name} is read in "
                f"{self.unit or 'no unit'}"
            )
        return True


@dataclass(frozen=True)
class WindFarm:
    """A farm of one export: the name its files carry and the turbines they name.

    Attributes:
        name: The farm as its archives and tables spell it.
        turbines: The turbines as the tables' names number them, in order.
    """

    name: str
    turbines: tuple[str, ...]


class GreenbyteCorpusReader:
    """Reads a wind farm's ten-minute turbine tables as the Greenbyte system exported them.

    A farm is a directory of ``<Farm>_SCADA_*.zip`` archives read compressed, each holding one
    table per turbine and period, ``Turbine_Data_<Farm>_<turbine>_<from>_-_<to>_<id>.csv``, beside
    status logs that are not read. A table opens with a preamble that names its time zone, which
    must be UTC, then a header and one row per ten minutes, stamped as ISO 8601 writes a date
    and a time; an empty cell or ``NaN`` is no reading.

    Of a table's three hundred columns the reader keeps the averages of the sixty signals the
    manufacturer's mapping names: what the turbine's controller sends. The minima, maxima and
    deviations beside them, and everything else the export holds or derives (lost production,
    availability, potential power), are not read. A column is a signal's by its title and its
    unit; the same title over another unit is refused. A turbine is a subset; every turbine of a
    farm is the same machine with the same sensors, so a channel is a signal and all turbines
    share it. Units are the calendar months in which a turbine sent a reading, keyed
    ``<turbine>/<YYYY-MM>``; the turbine stays the independent unit. An average is placed at the
    instant its row is stamped with, in hours since midnight UTC of 2016-01-01 on one axis for
    every farm. A turbine's first month opens at the whole hour of its first reading and its last
    closes at the whole hour after its last, so windows of whole hours lie on clock hours in
    every month.

    The later exports are cumulative: a table restarts its period every few hundred rows and
    repeats every instant before, with values in the newest rows alone, so an instant stands on
    up to ninety-two rows. The reader takes each signal's one value among the rows of an instant,
    whichever table they stand in, and refuses the data where two of them differ. Months named
    in ``excluded`` are left out of the units and the counts as if they were not there, and a
    name matching no month of a selected turbine is refused.

    The checksum covers a statement of the farm, the turbines selected and the signals, the months
    excluded where any are, and then every archive of the farm in name order. Every turbine read
    is kept in memory, since describing and then reading a publication would otherwise parse each
    table twice: about a quarter of a gigabyte a turbine.
    """

    KELMARSH: ClassVar[WindFarm] = WindFarm("Kelmarsh", ("1", "2", "3", "4", "5", "6"))
    PENMANSHIEL: ClassVar[WindFarm] = WindFarm(
        "Penmanshiel",
        ("01", "02", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"),
    )
    SIGNALS: ClassVar[tuple[Signal, ...]] = (
        Signal("Ambient temperature (converter)", "°C"),
        Signal("Apparent power", "kVA"),
        Signal("Blade angle (pitch position) A", "°"),
        Signal("Blade angle (pitch position) B", "°"),
        Signal("Blade angle (pitch position) C", "°"),
        Signal("CPU temperature", "°C"),
        Signal("Cable windings from calibration point", None),
        Signal("Current L1 / U", "A"),
        Signal("Current L2 / V", "A"),
        Signal("Current L3 / W", "A"),
        Signal("Drive train acceleration", "mm/s2"),
        Signal("Energy Export", "kWh"),
        Signal("Energy Import", "kWh"),
        Signal("Front bearing temperature", "°C"),
        Signal("Gear oil inlet pressure", "bar"),
        Signal("Gear oil inlet temperature", "°C"),
        Signal("Gear oil pump pressure", "bar"),
        Signal("Gear oil temperature", "°C"),
        Signal("Gearbox speed", "RPM"),
        Signal("Generator RPM", "RPM"),
        Signal("Generator bearing front temperature", "°C"),
        Signal("Generator bearing rear temperature", "°C"),
        Signal("Grid current", "A"),
        Signal("Grid frequency", "Hz"),
        Signal("Grid voltage", "V"),
        Signal("Hub temperature", "°C"),
        Signal("Metal particle count", None),
        Signal("Motor current axis 1", "A"),
        Signal("Motor current axis 2", "A"),
        Signal("Motor current axis 3", "A"),
        Signal("Nacelle ambient temperature", "°C"),
        Signal("Nacelle position", "°"),
        Signal("Nacelle temperature", "°C"),
        Signal("Power", "kW"),
        Signal("Power factor (cosphi)", None),
        Signal("Reactive Energy Export", "kvarh"),
        Signal("Reactive Energy Import", "kvarh"),
        Signal("Reactive power", "kvar"),
        Signal("Rear bearing temperature", "°C"),
        Signal("Rotor bearing temp", "°C"),
        Signal("Rotor speed", "RPM"),
        Signal("Stator temperature 1", "°C"),
        Signal("Temp. top box", "°C"),
        Signal("Temperature motor axis 1", "°C"),
        Signal("Temperature motor axis 2", "°C"),
        Signal("Temperature motor axis 3", "°C"),
        Signal("Tower Acceleration X", "mm/s2"),
        Signal("Tower Acceleration Y", "mm/s2"),
        Signal("Transformer cell temperature", "°C"),
        Signal("Transformer temperature", "°C"),
        Signal("Turbine Power setpoint", "kW"),
        Signal("Vane position 1+2", "°"),
        Signal("Voltage L1 / U", "V"),
        Signal("Voltage L2 / V", "V"),
        Signal("Voltage L3 / W", "V"),
        Signal("Wind direction", "°"),
        Signal("Wind speed", "m/s"),
        Signal("Wind speed Sensor 1", "m/s"),
        Signal("Wind speed Sensor 2", "m/s"),
        Signal("Yaw bearing angle", "°"),
    )

    def __init__(
        self,
        root: Path,
        farm: WindFarm,
        subsets: Iterable[str] = (),
        excluded: Iterable[str] = (),
    ) -> None:
        """Bind to the selected turbines of ``farm`` under ``root``; all of them unless named.

        Raises:
            ValueError: If a name is not a turbine of the farm, or an excluded month is not keyed
                ``<turbine>/<YYYY-MM>`` within a selected turbine.
        """
        self._root = root
        self._farm = farm
        self._turbines = chosen_subsets(tuple(subsets) or farm.turbines, farm.turbines, "turbine")
        self._excluded = ExcludedMonths(excluded, self._turbines, "turbine")
        self._loaded: dict[str, _Turbine] | None = None

    def describe(self) -> CorpusDescription:
        def contents() -> Iterator[bytes]:
            yield self._statement()
            yield self._excluded.statement()
            for path in self._archives():
                yield from chunks_of(path)

        # The checksum consumes the archives a chunk at a time, so none is ever held whole.
        checksum = Checksum.of_chunks(contents())
        kept = [
            (turbine, segment)
            for turbine in self._all().values()
            for segment in self._excluded.kept(turbine.name, turbine.months())
        ]
        return CorpusDescription(
            channel_schema=ChannelSchema(
                frozenset(Channel(signal.name, signal.unit) for signal in self.SIGNALS)
            ),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(kept),
                observation_count=sum(
                    turbine.observation_count_of(segment) for turbine, segment in kept
                ),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for turbine in self._all().values():
            for segment in self._excluded.kept(turbine.name, turbine.months()):
                yield CorpusUnit(
                    segment.key, TimeExtent(_hours(segment.start), _hours(segment.end))
                )

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        name = unit.part
        if name is None or name not in self._turbines:
            raise UnknownUnitError(f"{unit} is not a month of a selected turbine")
        turbine = self._all()[name]
        segment = turbine.segment_keyed(unit)
        if segment is None:
            raise UnknownUnitError(f"turbine {name} sent no reading in {unit.name}")
        if unit in self._excluded:
            raise UnknownUnitError(f"{unit} is excluded from the corpus")
        return turbine.observations_of(segment)

    def _statement(self) -> bytes:
        """The selection in words, the bytes the checksum opens with."""
        return (
            f"greenbyte: farm {self._farm.name}; turbines {','.join(self._turbines)}; "
            f"averages of {'|'.join(signal.name for signal in self.SIGNALS)}; "
            f"origin {ORIGIN_TEXT}\n"
        ).encode()

    def _archives(self) -> list[Path]:
        """The farm's archives in name order.

        Raises:
            CorpusDataNotFoundError: If the directory or its archives are missing.
        """
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        archives = sorted(self._root.glob(f"{self._farm.name}_SCADA_*.zip"))
        if not archives:
            raise CorpusDataNotFoundError(
                f"{self._root} holds no {self._farm.name}_SCADA_*.zip archive"
            )
        return archives

    def _all(self) -> dict[str, "_Turbine"]:
        """The selected turbines, assembled from the archives unless they were read before.

        The tables of every turbine lie interleaved in the archives, so one pass reads them all.

        Raises:
            CorpusDataNotFoundError: If a selected turbine has no table.
            MalformedCorpusDataError: If an archive is no archive, a table names another farm or
                a turbine the farm does not have, or a table violates its format.
        """
        if self._loaded is None:
            readings: dict[str, list[_Rows]] = {name: [] for name in self._turbines}
            for path in self._archives():
                try:
                    with zipfile.ZipFile(path) as archive:
                        for member in sorted(archive.namelist()):
                            turbine = self._turbine_of(path.name, member)
                            if turbine in readings:
                                readings[turbine].append(
                                    _Table(archive, member, self.SIGNALS).readings()
                                )
                except zipfile.BadZipFile as error:
                    raise MalformedCorpusDataError(
                        f"{path.name}: not a readable zip archive: {error}"
                    ) from error
            missing = [name for name, tables in readings.items() if not tables]
            if missing:
                raise CorpusDataNotFoundError(
                    f"{self._root} holds no table of {self._farm.name} turbines {missing}"
                )
            titles = [signal.name for signal in self.SIGNALS]
            self._loaded = {
                name: _Turbine.assemble(name, tables, titles) for name, tables in readings.items()
            }
        return self._loaded

    def _turbine_of(self, archive: str, member: str) -> str | None:
        """The turbine a member is a table of; ``None`` for a member that is no turbine's table."""
        match = _TABLE.fullmatch(Path(member).name)
        if match is None:
            return None
        if match["farm"] != self._farm.name:
            raise MalformedCorpusDataError(
                f"{archive}: {member} is a table of {match['farm']}, not of {self._farm.name}"
            )
        if match["turbine"] not in self._farm.turbines:
            raise MalformedCorpusDataError(
                f"{archive}: {member} names turbine {match['turbine']}, which "
                f"{self._farm.name} does not have"
            )
        return match["turbine"]


def _hours(instant: int) -> float:
    """An instant on the shared axis, by the arithmetic that places the observations too."""
    return (instant - ORIGIN_SECONDS) / _SECONDS_PER_HOUR


# The rows of a table that hold a reading: their instants in seconds since the epoch, and a
# column per signal, NaN where the row holds none of it.
type _Rows = tuple[NDArray[np.int64], NDArray[np.float64]]


@dataclass(frozen=True)
class _Table:
    """One turbine table inside an archive, parsed as a stream so that none is held whole."""

    archive: zipfile.ZipFile
    member: str
    signals: tuple[Signal, ...]

    def readings(self) -> _Rows:
        """The table's rows that hold a reading of a signal.

        Raises:
            MalformedCorpusDataError: If the preamble names no UTC time zone or no header
                follows it, the header is not UTF-8, repeats a column, holds none of a signal,
                holds one twice or states another unit for one, an instant is missing or no
                ISO 8601 stamp of a real date and time, or a value is no number or not finite.
        """
        # Imported here: the extra that provides pyarrow is needed by few readers, and a process
        # publishing another corpus must not need it.
        import pyarrow as pa
        import pyarrow.csv as pc

        skipped, header = self._header()
        columns = self._columns(header)
        named = list(columns.values())
        try:
            with self.archive.open(self.member) as stream:
                batches = pc.open_csv(
                    stream,
                    read_options=pc.ReadOptions(skip_rows=skipped),
                    convert_options=pc.ConvertOptions(
                        include_columns=[_TIME_COLUMN, *named],
                        column_types={
                            _TIME_COLUMN: pa.timestamp("s"),
                            **dict.fromkeys(named, pa.float64()),
                        },
                        null_values=["", "NaN"],
                    ),
                )
                rows = self._kept(batches, columns)
        # Whatever pyarrow cannot make of the bytes: a stamp that is no real instant, a cell
        # that is no number, a row of another width.
        except pa.ArrowException as error:
            raise MalformedCorpusDataError(f"{self.member}: {error}") from error
        return rows

    def _header(self) -> tuple[int, list[str]]:
        """How many lines the preamble holds, and the header that follows it."""
        zoned = False
        with self.archive.open(self.member) as stream:
            for skipped in range(_PREAMBLE_LIMIT):
                try:
                    line = stream.readline().decode("utf-8-sig").rstrip("\r\n")
                except UnicodeDecodeError as error:
                    # pyarrow names the columns by their bytes, so a header read leniently
                    # would name columns it then cannot find.
                    raise MalformedCorpusDataError(
                        f"{self.member}: line {skipped + 1} is not UTF-8"
                    ) from error
                if line.startswith(_TIME_COLUMN):
                    if not zoned:
                        raise MalformedCorpusDataError(
                            f"{self.member}: the preamble does not say {_TIME_ZONE!r}"
                        )
                    return skipped, next(csv.reader([line]))
                zoned = zoned or line.strip() == _TIME_ZONE
        raise MalformedCorpusDataError(
            f"{self.member}: no header opening with {_TIME_COLUMN!r} in the first "
            f"{_PREAMBLE_LIMIT} lines"
        )

    def _columns(self, header: list[str]) -> dict[int, str]:
        """The header's column that holds each signal's averages, by the signal's place."""
        repeated = sorted({column for column in header if header.count(column) > 1})
        if repeated:
            raise MalformedCorpusDataError(f"{self.member}: the header repeats {repeated}")
        columns = {}
        for index, signal in enumerate(self.signals):
            try:
                held = [column for column in header[1:] if signal.holds(column)]
            except ValueError as error:
                raise MalformedCorpusDataError(f"{self.member}: {error}") from error
            if len(held) > 1:
                # No table seen holds a signal twice; which column to believe is not ours to say.
                raise MalformedCorpusDataError(
                    f"{self.member}: the header holds {signal.name} in two columns, {held}"
                )
            if held:
                columns[index] = held[0]
        if not columns:
            raise MalformedCorpusDataError(f"{self.member}: the header holds no signal's column")
        return columns

    def _kept(self, batches: Any, columns: dict[int, str]) -> _Rows:
        """Every row holding a reading: its instant and one value per signal, NaN for none.

        pyarrow ships no type information, so a batch is whatever it hands back.
        """
        times: list[NDArray[np.int64]] = []
        values: list[NDArray[np.float64]] = []
        for batch in batches:
            stamps = batch.column(_TIME_COLUMN)
            if stamps.null_count:
                raise MalformedCorpusDataError(f"{self.member}: a row has no instant")
            block = np.full((batch.num_rows, len(self.signals)), np.nan)
            for index, column in columns.items():
                block[:, index] = batch.column(column).to_numpy(zero_copy_only=False)
            if np.isinf(block).any():
                raise MalformedCorpusDataError(f"{self.member}: non-finite value")
            held = np.isfinite(block).any(axis=1)
            times.append(stamps.to_numpy(zero_copy_only=False).astype(np.int64)[held])
            values.append(block[held])
        if not times:
            return np.zeros(0, dtype=np.int64), np.zeros((0, len(self.signals)))
        return np.concatenate(times), np.concatenate(values)


@dataclass(frozen=True)
class _Segment:
    """One calendar month in which a turbine sent a reading, in seconds since the epoch.

    ``rows`` are the turbine's rows the month holds, a half-open range.
    """

    key: UnitKey
    start: int
    end: int
    rows: tuple[int, int]


@dataclass(frozen=True)
class _Turbine:
    """A turbine's whole span: one row per instant it sent a reading at, a column per signal."""

    name: str
    signals: tuple[str, ...]
    times: NDArray[np.int64]
    values: NDArray[np.float64]
    segments: tuple[_Segment, ...]

    @classmethod
    def assemble(cls, name: str, tables: Iterable[_Rows], signals: Iterable[str]) -> "_Turbine":
        """Lay the tables' rows in time order with one row per instant, a column per signal.

        Raises:
            MalformedCorpusDataError: If the tables hold no reading, one precedes the origin,
                or two rows of one instant hold values of a signal that differ.
        """
        tables, titles = list(tables), tuple(signals)
        times = np.concatenate([instants for instants, _ in tables])
        values = np.concatenate([held for _, held in tables])
        if not len(times):
            raise MalformedCorpusDataError(f"turbine {name}: no reading of any signal")
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        if int(times[0]) < ORIGIN_SECONDS:
            raise MalformedCorpusDataError(f"turbine {name}: an instant precedes {ORIGIN_TEXT}")
        first = np.ones(len(times), dtype=bool)
        first[1:] = times[1:] != times[:-1]
        starts = np.flatnonzero(first)
        # The smallest and the largest value of an instant's rows, NaN aside, are one number
        # unless two rows disagree; all NaN, they are both NaN.
        low = np.fmin.reduceat(values, starts, axis=0)
        high = np.fmax.reduceat(values, starts, axis=0)
        if (low != high)[np.isfinite(low)].any():
            row, column = np.argwhere((low != high) & np.isfinite(low))[0]
            stamp = np.datetime64(int(times[starts[row]]), "s")
            raise MalformedCorpusDataError(
                f"turbine {name}: two rows of {stamp} disagree in {titles[int(column)]}"
            )
        instants = times[starts]
        return cls(name, titles, instants, low, cls._months(name, instants))

    @staticmethod
    def _months(name: str, times: NDArray[np.int64]) -> tuple[_Segment, ...]:
        """The months that hold a row, the first and last cut to the whole hours around the span."""
        opens = int(times[0]) // _SECONDS_PER_HOUR * _SECONDS_PER_HOUR
        closes = (int(times[-1]) // _SECONDS_PER_HOUR + 1) * _SECONDS_PER_HOUR
        stamps = times.astype("datetime64[s]")
        months = stamps.astype("datetime64[M]")
        present = np.unique(months)
        lower = np.searchsorted(months, present, side="left")
        upper = np.searchsorted(months, present, side="right")
        segments = []
        for month, first, beyond in zip(present, lower.tolist(), upper.tolist(), strict=True):
            start = int(month.astype("datetime64[s]").astype(np.int64))
            end = int((month + np.timedelta64(1, "M")).astype("datetime64[s]").astype(np.int64))
            segments.append(
                _Segment(
                    UnitKey.within(name, str(month)),
                    max(start, opens),
                    min(end, closes),
                    (first, beyond),
                )
            )
        return tuple(segments)

    def observation_count_of(self, segment: _Segment) -> int:
        first, beyond = segment.rows
        return int(np.isfinite(self.values[first:beyond]).sum())

    def months(self) -> dict[UnitKey, _Segment]:
        return {segment.key: segment for segment in self.segments}

    def segment_keyed(self, key: UnitKey) -> _Segment | None:
        return next((segment for segment in self.segments if segment.key == key), None)

    def observations_of(self, segment: _Segment) -> Iterator[Observation]:
        """The month's readings in time order, the signals of an instant in the policy's order."""
        first, beyond = segment.rows
        block = self.values[first:beyond]
        rows, columns = np.nonzero(np.isfinite(block))
        hours = ((self.times[first:beyond][rows] - ORIGIN_SECONDS) / _SECONDS_PER_HOUR).tolist()
        for column, time, value in zip(
            columns.tolist(), hours, block[rows, columns].tolist(), strict=True
        ):
            yield Observation(self.signals[column], time, value)
