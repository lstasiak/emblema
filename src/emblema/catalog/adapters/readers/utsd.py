import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    IndivisibleReadingError,
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

_SHARD_GLOB = "*.arrow"
_CORPUS_PREFIX = "utsd/"
_ITEM_ID = "item_id"
_TARGET = "target"
# The collection stores every value as a single-precision float; the checksum reads them as such.
_VALUE_DTYPE = "<f4"
# What a record is called: its dataset and the two numbers the publisher gave it.
_ITEM = re.compile(r"^(?P<dataset>.+)_(?P<a>\d+)_(?P<b>\d+)$")
_UNIT = re.compile(r"^series_\d+(_\d+)?$")
_VALUE_CHANNEL = Channel("value")
# Three digits order the variate names as the publisher numbers them, which is the order every
# consumer of a schema sees; no dataset of the collection has a thousand variates.
_VARIATE_WIDTH = 3


class UtsdLayout(StrEnum):
    """What the last two numbers of a record's name mean in one dataset of the collection.

    The collection stores every record as one univariate series named ``<dataset>_<a>_<b>``, but
    the publisher built the records of a multivariate dataset and of a collection of separate
    series differently, and only the dataset says which. Declared per dataset rather than inferred
    from the records, and checked against them when the dataset is described: it is a fact about
    the data, and how the data is read is a separate choice, the ``UtsdReading``.

    Attributes:
        SERIES_OF_VARIATES: ``a`` numbers a multivariate series and ``b`` its variate.
        VARIATES_OF_SERIES: ``a`` numbers the variate and ``b`` the series, as for a grid of
            locations each measuring the same quantities.
        COLLECTION_OF_SERIES: every record is a univariate series of its own; the two numbers
            only name it, since the publisher numbered such series in ``a``, in ``b``, or in
            ``b`` within blocks of a thousand numbered by ``a``.
    """

    SERIES_OF_VARIATES = "series_of_variates"
    VARIATES_OF_SERIES = "variates_of_series"
    COLLECTION_OF_SERIES = "collection_of_series"


class UtsdScale(StrEnum):
    """Which scale a record's values are read on.

    Attributes:
        WITHIN_UNIT: Each record's values less their mean, over their standard deviation, so a
            series is read on its own scale: the datasets gather series whose levels differ by
            orders of magnitude, and on one scale per channel most of them span a sliver of it
            while a few span the rest. A record that never varies is centred and not divided.
        AS_PUBLISHED: The values as the collection stores them.
    """

    WITHIN_UNIT = "within_unit"
    AS_PUBLISHED = "as_published"


@dataclass(frozen=True)
class UtsdReading:
    """How a dataset is read: whether its variates stay one unit's channels, and on what scale.

    A reading is the publication's choice, not a fact about the data, so it is part of what a
    description hashes: two readings of one dataset publish as two corpus versions.

    Attributes:
        independent_channels: Whether every record is a unit of its own on one channel, whatever
            the dataset's layout; a collection is read so either way.
        scale: The scale a record's values are read on.
    """

    independent_channels: bool = False
    scale: UtsdScale = UtsdScale.WITHIN_UNIT

    def tag(self) -> bytes:
        """The reading as the bytes a description's checksum begins with."""
        channels = "independent" if self.independent_channels else "as_laid_out"
        return f"reading:channels={channels};scale={self.scale}\0".encode()


@dataclass(frozen=True)
class UtsdDataset:
    """One dataset of the collection: the prefix its records carry and how they are laid out.

    Attributes:
        prefix: What every record of the dataset is named before its two numbers.
        layout: What those two numbers mean.
    """

    prefix: str
    layout: UtsdLayout

    @property
    def corpus_name(self) -> str:
        return f"{_CORPUS_PREFIX}{self.prefix}"


@dataclass(frozen=True, slots=True)
class _Record:
    """Where one record of the dataset lies in its shard, and the two numbers it is named by."""

    shard: int
    chunk: int
    index: int
    a: int
    b: int
    length: int


@dataclass(frozen=True)
class _Chunk:
    """One record batch of a shard as arrays: where each record's values start, and the values."""

    offsets: NDArray[np.int64]
    values: NDArray[np.float32]


class UtsdCorpusReader:
    """Reads one dataset of UTSD, the Unified Time Series Dataset, from its Arrow shards.

    The collection ships as shards of Arrow record batches, each row one univariate series:
    ``item_id`` names its dataset and two numbers, ``target`` holds its values, and the fields for
    a start, an end and a frequency are empty, so a value is placed at its index. Every dataset
    is a corpus of its own here — the datasets measure unrelated things — and a reader is bound
    to one, finding its records among the shards by name. What the two numbers mean is the
    dataset's declared ``UtsdLayout``; a dataset whose records contradict it is malformed. How
    the records become units, and on what scale, is the ``UtsdReading``.

    Values are evenly spaced, so the regime is regular; a unit of ``L`` values spans ``[0, L)``,
    the longest of its channels where they differ. A value stored as not-a-number is no
    observation. The checksum covers the reading and the dataset's records, name and stored values
    in the order of their two numbers, so a dataset describes the same wherever the shards cut it
    and differently under another reading. Shards stay mapped once read, and a record's values are
    a slice of its batch's, so no record is copied out of the map to be read.
    """

    CORPUS_PREFIX: ClassVar[str] = _CORPUS_PREFIX
    # The datasets of the 12G volume and their layouts, declared from the numbers their records
    # carry over the whole volume (2026-10-04) and checked against the shards when a dataset is
    # described.
    DATASETS: ClassVar[tuple[UtsdDataset, ...]] = (
        UtsdDataset("ERA5_pressure", UtsdLayout.VARIATES_OF_SERIES),
        UtsdDataset("ERA5_surface", UtsdLayout.VARIATES_OF_SERIES),
        UtsdDataset(
            "Energy_australian_electricity_demand_dataset", UtsdLayout.COLLECTION_OF_SERIES
        ),
        UtsdDataset(
            "Energy_london_smart_meters_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
        UtsdDataset("Environment_AustraliaRainfall", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Environment_BeijingPM25Quality", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Environment_BenzeneConcentration", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Health_AtrialFibrillation", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Health_BIDMC32HR", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Health_IEEEPPG", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Health_MotorImagery", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Health_PigArtPressure", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Health_PigCVP", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Health_SelfRegulationSCP1", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Health_SelfRegulationSCP2", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Health_TDBrain_csv", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("IoT_baian", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Nature_EigenWorms", UtsdLayout.SERIES_OF_VARIATES),
        UtsdDataset("Nature_Phoneme", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Nature_StarLightCurves", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset("Nature_Worms", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset(
            "Nature_kdd_cup_2018_dataset_without_missing_values", UtsdLayout.COLLECTION_OF_SERIES
        ),
        UtsdDataset(
            "Nature_temperature_rain_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
        UtsdDataset("Transport_pedestrian_counts_dataset", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset(
            "Web_kaggle_web_traffic_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
    )
    # Datasets of the volume no reading publishes, and why: a dataset of one univariate series is
    # one unit under every reading, and no split divides it into units to learn from and to hold
    # out.
    EXCLUDED: ClassVar[Mapping[str, str]] = {
        "Energy_wind_4_seconds_dataset": "one univariate series",
        "Nature_saugeenday_dataset": "one univariate series",
        "Nature_sunspot_dataset_without_missing_values": "one univariate series",
        "Nature_us_births_dataset": "one univariate series",
    }

    def __init__(
        self, root: Path, dataset: UtsdDataset, reading: UtsdReading | None = None
    ) -> None:
        self._root = root
        self._dataset = dataset
        self._reading = UtsdReading() if reading is None else reading
        self._units: dict[UnitKey, tuple[_Record, ...]] | None = None
        # pyarrow ships no type information, so a mapped shard is whatever it hands back.
        self._tables: dict[int, Any] = {}
        self._chunks: dict[tuple[int, int], _Chunk] = {}

    @classmethod
    def dataset_named(cls, corpus: str) -> UtsdDataset:
        """The declared dataset a corpus name stands for.

        Raises:
            ValueError: If the name is not that of a declared dataset of the collection.
        """
        for dataset in cls.DATASETS:
            if dataset.corpus_name == corpus:
                return dataset
        raise ValueError(f"no dataset of the collection is named {corpus!r}")

    def describe(self) -> CorpusDescription:
        units = self._indexed()
        records = sorted(
            (record for unit in units.values() for record in unit),
            key=lambda record: (record.a, record.b),
        )
        observations = 0

        def contents() -> Iterator[bytes]:
            nonlocal observations
            yield self._reading.tag()
            for record in records:
                values = self._values_of(record)
                observations += int(np.count_nonzero(np.isfinite(values)))
                yield self._name_of(record).encode("utf-8") + b"\0"
                yield values.tobytes()

        # The checksum consumes the records one at a time, so the dataset is never held whole.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(self._channel_of(record) for record in records)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum, unit_count=len(units), observation_count=observations
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for key, records in self._indexed().items():
            yield CorpusUnit(key, TimeExtent(0.0, float(max(record.length for record in records))))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        if not _UNIT.match(unit.value):
            raise UnknownUnitError(f"{unit} is not a series of the dataset")
        records = self._indexed().get(unit)
        if records is None:
            raise UnknownUnitError(f"{self._dataset.prefix} has no {unit}")
        return self._observations_of(records)

    def _observations_of(self, records: tuple[_Record, ...]) -> Iterator[Observation]:
        """The unit's values index by index, its channels in variate order at each."""
        channels = [
            (self._channel_of(record).name, self._read(record).tolist()) for record in records
        ]
        for index in range(max(len(values) for _, values in channels)):
            for channel, values in channels:
                if index < len(values) and math.isfinite(values[index]):
                    yield Observation(channel, float(index), values[index])

    def _read(self, record: _Record) -> NDArray[np.float64]:
        """The record's values on the reading's scale, not-a-number where none was stored."""
        values = self._values_of(record).astype(np.float64)
        finite = values[np.isfinite(values)]
        if self._reading.scale is UtsdScale.AS_PUBLISHED or finite.size == 0:
            return values
        centred = values - finite.mean()
        spread = float(finite.std())
        return centred / spread if spread > 0.0 else centred

    @property
    def _one_unit_per_record(self) -> bool:
        return (
            self._reading.independent_channels
            or self._dataset.layout is UtsdLayout.COLLECTION_OF_SERIES
        )

    def _channel_of(self, record: _Record) -> Channel:
        if self._one_unit_per_record:
            return _VALUE_CHANNEL
        return Channel(f"variate_{self._variate_of(record):0{_VARIATE_WIDTH}d}")

    def _series_of(self, record: _Record) -> int:
        return record.b if self._dataset.layout is UtsdLayout.VARIATES_OF_SERIES else record.a

    def _variate_of(self, record: _Record) -> int:
        return record.a if self._dataset.layout is UtsdLayout.VARIATES_OF_SERIES else record.b

    def _name_of(self, record: _Record) -> str:
        return f"{self._dataset.prefix}_{record.a}_{record.b}"

    def _indexed(self) -> dict[UnitKey, tuple[_Record, ...]]:
        """The dataset's records by unit, found on first use by scanning every shard's names."""
        if self._units is None:
            self._units = self._laid_out(self._scan())
        return self._units

    def _shards(self) -> list[Path]:
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        shards = sorted(self._root.glob(_SHARD_GLOB), key=lambda path: path.name)
        if not shards:
            raise CorpusDataNotFoundError(f"{self._root} holds no shard")
        return shards

    def _scan(self) -> list[_Record]:
        """Find the dataset's records by name, touching no values.

        Raises:
            CorpusDataNotFoundError: If the shards hold no record of the dataset.
            MalformedCorpusDataError: If a shard is not one of the collection's, or a record's
                name is not a dataset's and two numbers, or a record holds no value.
        """
        found: list[_Record] = []
        prefix = f"{self._dataset.prefix}_"
        for shard, path in enumerate(self._shards()):
            table = self._table(shard)
            row = 0
            batches = zip(table.column(_ITEM_ID).chunks, table.column(_TARGET).chunks, strict=True)
            for chunk, (names, targets) in enumerate(batches):
                # The lengths come from the list offsets, so no value of any record is touched.
                lengths = targets.value_lengths().to_pylist()
                for index, (name, length) in enumerate(
                    zip(names.to_pylist(), lengths, strict=True)
                ):
                    if not name.startswith(prefix):
                        continue
                    where = f"{path.name}, row {row + index}"
                    parsed = _ITEM.match(name)
                    if parsed is None:
                        raise MalformedCorpusDataError(
                            f"{where}: {name!r} does not name a dataset and two numbers"
                        )
                    # A dataset whose name extends this one's is another dataset, not a bad one.
                    if parsed["dataset"] != self._dataset.prefix:
                        continue
                    if length == 0:
                        raise MalformedCorpusDataError(f"{where}: {name!r} holds no value")
                    found.append(
                        _Record(shard, chunk, index, int(parsed["a"]), int(parsed["b"]), length)
                    )
                row += len(names)
        if not found:
            raise CorpusDataNotFoundError(f"{self._root} holds no record of {self._dataset.prefix}")
        return found

    def _laid_out(self, found: list[_Record]) -> dict[UnitKey, tuple[_Record, ...]]:
        """The records grouped into units as the layout and the reading say, checked.

        Raises:
            MalformedCorpusDataError: If a record is stored twice, or the series of a
                multivariate layout do not all carry the same variates.
            IndivisibleReadingError: If the reading leaves the dataset a single unit.
        """
        name = self._dataset.prefix
        found.sort(key=lambda record: (record.a, record.b))
        for earlier, later in pairwise(found):
            if (earlier.a, earlier.b) == (later.a, later.b):
                raise MalformedCorpusDataError(f"{name}_{later.a}_{later.b} is stored twice")
        units: dict[UnitKey, tuple[_Record, ...]] = {}
        if self._one_unit_per_record:
            for record in found:
                units[UnitKey(f"series_{record.a}_{record.b}")] = (record,)
        else:
            for record in sorted(found, key=lambda r: (self._series_of(r), self._variate_of(r))):
                key = UnitKey(f"series_{self._series_of(record)}")
                units[key] = (*units.get(key, ()), record)
            carried = {
                tuple(self._variate_of(record) for record in records) for records in units.values()
            }
            if len(carried) > 1:
                raise MalformedCorpusDataError(
                    f"{name} is declared {self._dataset.layout}, but its series do not all carry "
                    f"the same variates: {sorted(len(variates) for variates in carried)} per "
                    "series"
                )
        if len(units) == 1:
            raise IndivisibleReadingError(
                f"{name} read so is one unit, which no split divides; read its channels "
                "independently"
            )
        return units

    def _values_of(self, record: _Record) -> NDArray[np.float32]:
        """The record's values as the collection stores them: a slice of its batch, not a copy."""
        chunk = self._chunk(record.shard, record.chunk)
        return chunk.values[chunk.offsets[record.index] : chunk.offsets[record.index + 1]]

    def _chunk(self, shard: int, number: int) -> _Chunk:
        """One batch of a shard as arrays, built once: the offsets and the values they index.

        The values are read over the map where they hold no null, and copied with not-a-number
        where one does; the offsets index the whole child array, as a sliced batch's still do.
        """
        if (shard, number) not in self._chunks:
            targets = self._table(shard).column(_TARGET).chunk(number)
            self._chunks[(shard, number)] = _Chunk(
                offsets=np.asarray(targets.offsets.to_numpy(), dtype=np.int64),
                values=np.asarray(
                    targets.values.to_numpy(zero_copy_only=False), dtype=_VALUE_DTYPE
                ),
            )
        return self._chunks[(shard, number)]

    def _table(self, shard: int) -> Any:
        """The shard mapped as a table, mapped once: units of a dataset may interleave shards."""
        if shard not in self._tables:
            self._tables[shard] = self._table_of(self._shards()[shard])
        return self._tables[shard]

    @staticmethod
    def _table_of(path: Path) -> Any:
        """The shard mapped as a table, checked to be one of the collection's.

        Raises:
            MalformedCorpusDataError: If the file is not an Arrow stream of named float series.
        """
        # Imported here: the extra that provides Arrow is needed by this reader alone, and a
        # process publishing another corpus must not need it.
        import pyarrow as pa
        import pyarrow.ipc as ipc

        try:
            table = ipc.open_stream(pa.memory_map(str(path))).read_all()
        except pa.ArrowInvalid as error:
            raise MalformedCorpusDataError(f"{path.name}: not an Arrow stream: {error}") from error
        schema = table.schema
        if (
            _ITEM_ID not in schema.names
            or _TARGET not in schema.names
            or not pa.types.is_string(schema.field(_ITEM_ID).type)
            or not pa.types.is_list(schema.field(_TARGET).type)
            or not pa.types.is_float32(schema.field(_TARGET).type.value_type)
        ):
            raise MalformedCorpusDataError(
                f"{path.name}: expected named series of single-precision values, got {schema}"
            )
        return table
