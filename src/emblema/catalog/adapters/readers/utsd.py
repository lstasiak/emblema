import math
import re
from collections.abc import Iterator
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
# What a record is called: its dataset, the series it belongs to and the variate it carries.
_ITEM = re.compile(r"^(?P<dataset>.+)_(?P<series>\d+)_(?P<variate>\d+)$")
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
    from the records, and checked against them when the dataset is described.

    Attributes:
        SERIES_OF_VARIATES: ``a`` numbers a multivariate series and ``b`` its variate: the series
            is the unit, each variate a channel, and every series carries the same variates.
        COLLECTION_OF_SERIES: every record is a univariate series of its own, a unit on the
            collection's single channel; the two numbers only name it, since the publisher
            numbered such series in ``a``, in ``b``, or in ``b`` within blocks of a thousand
            numbered by ``a``.
    """

    SERIES_OF_VARIATES = "series_of_variates"
    COLLECTION_OF_SERIES = "collection_of_series"


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
    """Where one record of the dataset lies and what it names."""

    shard: int
    row: int
    series: int
    variate: int
    length: int


@dataclass(frozen=True)
class _Index:
    """Every record of the dataset, found once by scanning the names of every shard.

    Attributes:
        units: Each unit's records in variate order, by unit key, in the order units are read.
        variates: Every variate number the dataset's records carry.
    """

    units: dict[UnitKey, tuple[_Record, ...]]
    variates: frozenset[int]


class UtsdCorpusReader:
    """Reads one dataset of UTSD, the Unified Time Series Dataset, from its Arrow shards.

    The collection ships as shards of Arrow record batches, each row one univariate series:
    ``item_id`` names its dataset and two numbers, ``target`` holds its values, and the fields for
    a start, an end and a frequency are empty, so a value is placed at its index. Every dataset
    is a corpus of its own here — the datasets measure unrelated things — and a reader is bound
    to one, finding its records among the shards by name. What the two numbers mean is the
    dataset's declared ``UtsdLayout``; a dataset whose records contradict it is malformed.

    Values are evenly spaced, so the regime is regular; a unit of ``L`` values spans ``[0, L)``,
    the longest of its variates where they differ. A value stored as not-a-number is no
    observation. The checksum covers the dataset's records alone, name and values in canonical
    order, so a dataset describes the same wherever the shards cut it. Shards stay mapped once
    read, so the records of a dataset are served without reopening a shard per unit.
    """

    CORPUS_PREFIX: ClassVar[str] = _CORPUS_PREFIX
    # The datasets of the 12G volume and their layouts, declared from the names their records
    # carry (sampled through the repository's rows API on 2026-10-04) and checked against the
    # shards when a dataset is described. A dataset the shards hold and this table does not is
    # found by the test that lists them where the shards are.
    DATASETS: ClassVar[tuple[UtsdDataset, ...]] = (
        UtsdDataset(
            "Energy_london_smart_meters_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
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
            "Nature_temperature_rain_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
        UtsdDataset("Transport_pedestrian_counts_dataset", UtsdLayout.COLLECTION_OF_SERIES),
        UtsdDataset(
            "Web_kaggle_web_traffic_dataset_without_missing_values",
            UtsdLayout.COLLECTION_OF_SERIES,
        ),
    )

    def __init__(self, root: Path, dataset: UtsdDataset) -> None:
        self._root = root
        self._dataset = dataset
        self._index: _Index | None = None
        # pyarrow ships no type information, so a mapped shard is whatever it hands back.
        self._tables: dict[int, Any] = {}

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
        index = self._indexed()
        observations = 0

        def contents() -> Iterator[bytes]:
            nonlocal observations
            for records in index.units.values():
                for record in records:
                    values = self._values_of(record)
                    observations += int(np.count_nonzero(np.isfinite(values)))
                    yield self._name_of(record).encode("utf-8") + b"\0"
                    yield values.tobytes()

        # The checksum consumes the records one at a time, so the dataset is never held whole.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=self._schema(index),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum, unit_count=len(index.units), observation_count=observations
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for key, records in self._indexed().units.items():
            yield CorpusUnit(key, TimeExtent(0.0, float(max(record.length for record in records))))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        if not _UNIT.match(unit.value):
            raise UnknownUnitError(f"{unit} is not a series of the dataset")
        records = self._indexed().units.get(unit)
        if records is None:
            raise UnknownUnitError(f"{self._dataset.prefix} has no {unit}")
        return self._observations_of(records)

    def _observations_of(self, records: tuple[_Record, ...]) -> Iterator[Observation]:
        """The unit's values index by index, its channels in variate order at each."""
        channels = [(self._channel_of(record).name, self._values_of(record)) for record in records]
        for index in range(max(len(values) for _, values in channels)):
            for channel, values in channels:
                if index < len(values) and math.isfinite(values[index]):
                    yield Observation(channel, float(index), float(values[index]))

    def _schema(self, index: _Index) -> ChannelSchema:
        if self._dataset.layout is UtsdLayout.COLLECTION_OF_SERIES:
            return ChannelSchema(frozenset((_VALUE_CHANNEL,)))
        return ChannelSchema(frozenset(_variate_channel(variate) for variate in index.variates))

    def _channel_of(self, record: _Record) -> Channel:
        if self._dataset.layout is UtsdLayout.COLLECTION_OF_SERIES:
            return _VALUE_CHANNEL
        return _variate_channel(record.variate)

    def _name_of(self, record: _Record) -> str:
        return f"{self._dataset.prefix}_{record.series}_{record.variate}"

    def _indexed(self) -> _Index:
        """The dataset's records, found on first use by scanning every shard's names."""
        if self._index is None:
            self._index = self._scan()
        return self._index

    def _shards(self) -> list[Path]:
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        shards = sorted(self._root.glob(_SHARD_GLOB), key=lambda path: path.name)
        if not shards:
            raise CorpusDataNotFoundError(f"{self._root} holds no shard")
        return shards

    def _scan(self) -> _Index:
        """Find the dataset's records by name, touching no values.

        Raises:
            CorpusDataNotFoundError: If the shards hold no record of the dataset.
            MalformedCorpusDataError: If a shard is not one of the collection's, a record's name
                is not a dataset's and two numbers, or the records contradict the layout.
        """
        found: list[_Record] = []
        prefix = f"{self._dataset.prefix}_"
        for shard, path in enumerate(self._shards()):
            table = self._table_of(path)
            names = table.column(_ITEM_ID).to_pylist()
            # The lengths come from the list offsets, so no value of any record is touched here.
            lengths = [
                length
                for chunk in table.column(_TARGET).chunks
                for length in chunk.value_lengths().to_pylist()
            ]
            for row, (name, length) in enumerate(zip(names, lengths, strict=True)):
                if not name.startswith(prefix):
                    continue
                parsed = _ITEM.match(name)
                if parsed is None:
                    raise MalformedCorpusDataError(
                        f"{path.name}, row {row}: {name!r} does not name a series and a variate"
                    )
                # A dataset whose name extends this one's is another dataset, not a bad record.
                if parsed["dataset"] != self._dataset.prefix:
                    continue
                if length == 0:
                    raise MalformedCorpusDataError(
                        f"{path.name}, row {row}: {name!r} holds no value"
                    )
                found.append(
                    _Record(shard, row, int(parsed["series"]), int(parsed["variate"]), length)
                )
        if not found:
            raise CorpusDataNotFoundError(f"{self._root} holds no record of {self._dataset.prefix}")
        return self._laid_out(found)

    def _laid_out(self, found: list[_Record]) -> _Index:
        """The records grouped into units as the layout says, checked against it."""
        name = self._dataset.prefix
        found.sort(key=lambda found_record: (found_record.series, found_record.variate))
        for earlier, later in pairwise(found):
            if (earlier.series, earlier.variate) == (later.series, later.variate):
                raise MalformedCorpusDataError(
                    f"{name}: series {later.series} carries variate {later.variate} twice"
                )
        units: dict[UnitKey, tuple[_Record, ...]] = {}
        if self._dataset.layout is UtsdLayout.COLLECTION_OF_SERIES:
            for record in found:
                units[UnitKey(f"series_{record.series}_{record.variate}")] = (record,)
        else:
            for record in found:
                key = UnitKey(f"series_{record.series}")
                units[key] = (*units.get(key, ()), record)
            variates = {tuple(record.variate for record in records) for records in units.values()}
            if len(variates) > 1:
                raise MalformedCorpusDataError(
                    f"{name} is declared series of variates, but its series do not all carry "
                    f"the same variates: {sorted(len(carried) for carried in variates)} per series"
                )
        return _Index(units, frozenset(record.variate for record in found))

    def _values_of(self, record: _Record) -> NDArray[np.float32]:
        """The record's values as the collection stores them, single-precision floats."""
        column = self._table(record.shard).column(_TARGET)[record.row]
        return np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=_VALUE_DTYPE)

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


def _variate_channel(variate: int) -> Channel:
    return Channel(f"variate_{variate:0{_VARIATE_WIDTH}d}")
