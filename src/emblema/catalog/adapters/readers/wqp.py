import csv
import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    InvalidUnitKeyError,
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

_BATCHES = "batches"
_BATCH_PATTERN = "batch-*.zip"
_SECONDS_PER_HOUR = 3600.0
_FIRST = datetime(2010, 1, 1, tzinfo=UTC)
_END = datetime(2024, 1, 1, tzinfo=UTC)
# Every location shares the fetched years as its extent, so windows of a fixed length cut every
# location alike; a reading a zone moves past either edge lies outside the years and is dropped.
_YEARS = TimeExtent(_FIRST.timestamp() / _SECONDS_PER_HOUR, _END.timestamp() / _SECONDS_PER_HOUR)
_WATER = "water"
_QUALITY_CONTROL = "Quality Control"
# A result whose statistical base is any other than these is computed from other results (a mean,
# a daily maximum, a ratio, a recovery) rather than measured.
_SINGLE_MEASUREMENTS = frozenset({"", "mpn", "instantaneous"})
_CLOCK = re.compile(r"([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
# Fractions that name the water as taken, which the portal's organisations label inconsistently;
# a reading of one of them comes before a dissolved or other fraction at the same instant.
_WHOLE_WATER = frozenset({"", "TOTAL", "UNFILTERED", "NONE"})
_COLUMNS = (
    "OrganizationIdentifier",
    "MonitoringLocationIdentifier",
    "ProviderName",
    "ActivityIdentifier",
    "ActivityMediaName",
    "ActivityTypeCode",
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "ActivityStartTime/TimeZoneCode",
    "ActivityDepthHeightMeasure/MeasureValue",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "StatisticalBaseCode",
    "ResultDepthHeightMeasure/MeasureValue",
)
# Offsets from UTC, in hours, of the zone codes the portal's activities carry; a time under any
# other code, or none, is read as UTC.
_ZONES: Mapping[str, float] = {
    "UTC": 0.0,
    "GMT": 0.0,
    "AST": -4.0,
    "ADT": -3.0,
    "EST": -5.0,
    "EDT": -4.0,
    "CST": -6.0,
    "CDT": -5.0,
    "MST": -7.0,
    "MDT": -6.0,
    "PST": -8.0,
    "PDT": -7.0,
    "AKST": -9.0,
    "AKDT": -8.0,
    "HST": -10.0,
    "HAST": -10.0,
    "HADT": -9.0,
    "SST": -11.0,
    "GST": 10.0,
    "CHST": 10.0,
}


@dataclass(frozen=True)
class _Reading:
    """One result that counts as a reading, and what decides between readings of one instant.

    Attributes:
        organisation: Who reported it.
        activity: The activity it came from.
        name: Its characteristic.
        fraction: The sample fraction as published.
        unit: Its unit code, case folded.
        spelling: Its unit code as published.
        depth: Its depth as published, the result's before the activity's; empty where none.
        time: Hours since the epoch, UTC.
        value: The number.
        row: Its row in the file, which breaks the last tie.
    """

    organisation: str
    activity: str
    name: str
    fraction: str
    unit: str
    spelling: str
    depth: str
    time: float
    value: float
    row: int

    def precedence(self) -> tuple[int, int, float, int]:
        """The order in which readings of one channel at one instant are preferred."""
        whole = 0 if self.fraction.strip().upper() in _WHOLE_WATER else 1
        if not self.depth:
            return whole, 0, 0.0, self.row
        try:
            depth = float(self.depth)
        except ValueError:
            return whole, 2, 0.0, self.row
        return whole, 1, depth, self.row


@dataclass(frozen=True)
class _Index:
    """What a pass over every batch decides: the channels kept and the locations that hold them.

    Attributes:
        channels: Each kept characteristic by name, with the unit its readings are taken in,
            case folded.
        spellings: The unit of each kept characteristic as most of its readings spell it.
        batches: The batch file of each kept location, batch by batch and in key order within one.
        counts: The observations each kept location holds.
    """

    channels: Mapping[str, str]
    spellings: Mapping[str, str]
    batches: Mapping[UnitKey, Path]
    counts: Mapping[UnitKey, int]


class WaterQualityPortalCorpusReader:
    """Reads the US Water Quality Portal's results of stream, lake and reservoir locations.

    The root holds ``batches/batch-<n>.zip``, each the portal's narrow result CSV for a batch of
    locations over 2010 to 2023, as ``scripts/wqp_fetch.py`` fetched them. A location is a unit,
    keyed ``<provider>/<location identifier>`` (an identifier may hold a slash of its own), with
    the fetched years as its extent; a characteristic is a channel (ADR-0065).

    A result is a reading when it was taken in water by the organisation that reports most of the
    location's results (every row counted), is not a quality-control sample or a statistic of other
    results, gives a time of day and a finite number. A result dated without a time is dropped, and
    so are the readings of an activity that reports one characteristic, fraction, unit and depth
    more than once: neither carries an instant of its own, and a false instant would hand a model a
    clock. Of the readings of one characteristic and unit at one instant, one is kept: the water as
    taken before a dissolved fraction, then the shallowest, then the first in the file, since a
    depth profile or a repeat says nothing about when the channel is measured.

    Each characteristic is read in its most frequent unit, codes compared regardless of case and
    spelt as most of its readings spell it; results in other units are dropped, since a channel's
    values are normalised together. Names are kept as published, fractions not told apart. The
    ``channels`` characteristics measured at the most locations of all fetched are kept, so that
    every channel is measured widely enough to be normalised on any side of a split, and a
    location enters with at least ``least`` readings on them. Times are hours since the epoch, UTC.

    The checksum covers the batch files in name order. The batches are indexed when the corpus is
    described or first read, and the batch last read is kept, so its locations are served from
    memory.
    """

    CHANNELS: ClassVar[int] = 64
    LEAST: ClassVar[int] = 50

    def __init__(self, root: Path, channels: int = CHANNELS, least: int = LEAST) -> None:
        if channels < 1 or least < 1:
            raise ValueError(f"channels and least must be positive, got {channels} and {least}")
        self._root = root
        self._channels = channels
        self._least = least
        self._index: _Index | None = None
        self._loaded: tuple[Path, dict[UnitKey, list[Observation]]] | None = None

    def describe(self) -> CorpusDescription:
        paths = self._batch_paths()
        checksum = Checksum.of_chunks(path.read_bytes() for path in paths)
        # Indexed afresh, so the counts and the schema describe the bytes the checksum covers.
        self._index = self._index_of(paths)
        self._loaded = None
        index = self._index
        return CorpusDescription(
            channel_schema=ChannelSchema(
                frozenset(Channel(name, unit or None) for name, unit in index.spellings.items())
            ),
            sampling_regime=SamplingRegime.IRREGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(index.counts),
                observation_count=sum(index.counts.values()),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for key in self._indexed().batches:
            yield CorpusUnit(key, _YEARS)

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        path = self._indexed().batches.get(unit)
        if path is None:
            raise UnknownUnitError(f"{unit} is not a location the corpus keeps")
        return iter(self._batch(path)[unit])

    def _batch_paths(self) -> list[Path]:
        paths = sorted((self._root / _BATCHES).glob(_BATCH_PATTERN))
        if not paths:
            raise CorpusDataNotFoundError(
                f"no {_BATCH_PATTERN} under {self._root / _BATCHES}; "
                "fetch them with scripts/wqp_fetch.py"
            )
        return paths

    def _indexed(self) -> _Index:
        if self._index is None:
            self._index = self._index_of(self._batch_paths())
        return self._index

    def _index_of(self, paths: list[Path]) -> _Index:
        """One pass over every batch: units of measure, the channels kept, the locations kept."""
        found: dict[UnitKey, Path] = {}
        held: dict[UnitKey, Counter[tuple[str, str]]] = {}
        spelt: Counter[tuple[str, str, str]] = Counter()
        for path in paths:
            for key, readings in self._readings_of(path).items():
                if key in found:
                    raise MalformedCorpusDataError(
                        f"{path.name}: {key} was already read from {found[key].name}"
                    )
                found[key] = path
                held[key] = Counter((r.name, r.unit) for r in readings)
                spelt.update((r.name, r.unit, r.spelling) for r in readings)
        totals: Counter[tuple[str, str]] = Counter()
        for location in held.values():
            totals.update(location)
        by_name: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (name, unit), count in totals.items():
            by_name[name].append((unit, count))
        # The most frequent unit, ties to the first code, so a rerun reads alike.
        units = {
            name: min(found_units, key=lambda pair: (-pair[1], pair[0]))[0]
            for name, found_units in by_name.items()
        }
        breadth = Counter(
            name for location in held.values() for name, unit in location if units[name] == unit
        )
        kept = sorted(breadth, key=lambda name: (-breadth[name], name))[: self._channels]
        channels = {name: units[name] for name in sorted(kept)}
        counts = {
            key: sum(n for (name, unit), n in readings.items() if channels.get(name) == unit)
            for key, readings in held.items()
        }
        # Batch by batch, so that reading every unit in turn parses each batch once.
        chosen = sorted(
            (key for key, n in counts.items() if n >= self._least),
            key=lambda key: (found[key].name, str(key)),
        )
        spellings = {
            name: min(
                (
                    (spelling, n)
                    for (named, unit, spelling), n in spelt.items()
                    if named == name and unit == channels[name]
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )[0]
            for name in channels
        }
        return _Index(
            channels=channels,
            spellings=spellings,
            batches={key: found[key] for key in chosen},
            counts={key: counts[key] for key in chosen},
        )

    def _batch(self, path: Path) -> dict[UnitKey, list[Observation]]:
        """The kept locations' observations of one batch, in time order, the last batch kept."""
        if self._loaded is None or self._loaded[0] != path:
            index = self._indexed()
            observations = {
                key: sorted(
                    (
                        Observation(r.name, r.time, r.value)
                        for r in readings
                        if index.channels.get(r.name) == r.unit
                    ),
                    key=lambda o: (o.time, o.channel),
                )
                for key, readings in self._readings_of(path).items()
                if key in index.batches
            }
            self._loaded = (path, observations)
        return self._loaded[1]

    @classmethod
    def _readings_of(cls, path: Path) -> dict[UnitKey, list[_Reading]]:
        """Every location's readings in one batch, one per characteristic, unit and instant.

        Raises:
            MalformedCorpusDataError: If the file is not a zip of one CSV member holding the
                columns read, or a row names no location or no instant.
        """
        candidates: dict[UnitKey, list[_Reading]] = defaultdict(list)
        reported: dict[UnitKey, Counter[str]] = defaultdict(Counter)
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                if len(members) != 1:
                    raise MalformedCorpusDataError(
                        f"{path.name}: expected an archive of one member, got {members}"
                    )
                with archive.open(members[0]) as member:
                    text = io.TextIOWrapper(member, encoding="utf-8-sig", newline="")
                    table = csv.DictReader(text, restval="")
                    absent = [c for c in _COLUMNS if c not in (table.fieldnames or ())]
                    if absent:
                        raise MalformedCorpusDataError(f"{path.name}: no column {absent}")
                    for row in table:
                        where = f"{path.name}, line {table.line_num}"
                        key = cls._key_of(where, row)
                        reported[key][row["OrganizationIdentifier"]] += 1
                        reading = cls._reading_of(where, table.line_num, row)
                        if reading is not None:
                            candidates[key].append(reading)
        except zipfile.BadZipFile as error:
            raise MalformedCorpusDataError(f"{path.name}: not a zip archive") from error
        except (UnicodeDecodeError, csv.Error) as error:
            raise MalformedCorpusDataError(
                f"{path.name}: not a UTF-8 CSV table: {error}"
            ) from error
        return {
            key: cls._kept(candidates.get(key, []), cls._main(counts))
            for key, counts in reported.items()
        }

    @staticmethod
    def _main(counts: Counter[str]) -> str:
        """The organisation reporting most, ties to the first name, so a rerun reads alike."""
        return min(counts, key=lambda name: (-counts[name], name))

    @staticmethod
    def _kept(readings: Sequence[_Reading], organisation: str) -> list[_Reading]:
        """A location's readings through one organisation, without repeats or false instants."""
        mine = [r for r in readings if r.organisation == organisation]
        repeats = Counter((r.activity, r.name, r.fraction, r.unit, r.depth) for r in mine)
        timed = [r for r in mine if repeats[(r.activity, r.name, r.fraction, r.unit, r.depth)] == 1]
        chosen: dict[tuple[str, str, float], _Reading] = {}
        for r in timed:
            slot = (r.name, r.unit, r.time)
            if slot not in chosen or r.precedence() < chosen[slot].precedence():
                chosen[slot] = r
        return sorted(chosen.values(), key=lambda r: r.row)

    @staticmethod
    def _key_of(where: str, row: Mapping[str, str]) -> UnitKey:
        """The unit a result row belongs to.

        Raises:
            MalformedCorpusDataError: If the row names no database or location.
        """
        try:
            return UnitKey.within(row["ProviderName"], row["MonitoringLocationIdentifier"])
        except InvalidUnitKeyError as error:
            raise MalformedCorpusDataError(f"{where}: {error}") from error

    @staticmethod
    def _reading_of(where: str, line: int, row: Mapping[str, str]) -> _Reading | None:
        """The reading a result row is, or ``None`` where it is none.

        Raises:
            MalformedCorpusDataError: If its date or time of day is not one.
        """
        if row["ActivityMediaName"].strip().casefold() != _WATER:
            return None
        if row["ActivityTypeCode"].startswith(_QUALITY_CONTROL):
            return None
        if row["StatisticalBaseCode"].strip().casefold() not in _SINGLE_MEASUREMENTS:
            return None
        clock = row["ActivityStartTime/Time"].strip()
        if not clock:
            return None
        try:
            value = float(row["ResultMeasureValue"])
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        stamp = _CLOCK.fullmatch(clock)
        if stamp is None:
            raise MalformedCorpusDataError(f"{where}: {clock!r} is not a time of day")
        day = row["ActivityStartDate"].strip()
        zone = row["ActivityStartTime/TimeZoneCode"].strip().upper()
        try:
            if not _DATE.fullmatch(day):
                raise ValueError(day)
            hours, minutes, seconds = (int(part or 0) for part in stamp.groups())
            taken = date.fromisoformat(day)
            moment = datetime(
                taken.year, taken.month, taken.day, hours, minutes, seconds, tzinfo=UTC
            ) - timedelta(hours=_ZONES.get(zone, 0.0))
        except (ValueError, OverflowError) as error:
            raise MalformedCorpusDataError(
                f"{where}: {day!r} {clock!r} is not an instant"
            ) from error
        time = moment.timestamp() / _SECONDS_PER_HOUR
        characteristic = row["CharacteristicName"].strip()
        if not characteristic or not _YEARS.contains(time):
            return None
        code = row["ResultMeasure/MeasureUnitCode"].strip()
        return _Reading(
            organisation=row["OrganizationIdentifier"],
            activity=row["ActivityIdentifier"],
            name=characteristic,
            fraction=row["ResultSampleFractionText"],
            unit=code.casefold(),
            spelling=code,
            depth=(
                row["ResultDepthHeightMeasure/MeasureValue"]
                or row["ActivityDepthHeightMeasure/MeasureValue"]
            ).strip(),
            time=time,
            value=value,
            row=line,
        )
