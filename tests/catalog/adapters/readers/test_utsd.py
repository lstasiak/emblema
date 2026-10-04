import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from emblema.catalog.adapters.readers.utsd import UtsdCorpusReader, UtsdDataset, UtsdLayout
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.corpora import raw_root, sample

pa = pytest.importorskip("pyarrow")
ipc = pytest.importorskip("pyarrow.ipc")

SAMPLE = sample("utsd") / "UTSD-12G"
SCP1 = UtsdCorpusReader.dataset_named("utsd/Health_SelfRegulationSCP1")
RAIN = UtsdCorpusReader.dataset_named("utsd/Nature_temperature_rain_dataset_without_missing_values")
# What the sample holds: three six-variate series of the brain-signal dataset, cut to 32 values,
# the third of them spanning both shards; and five series of the rainfall collection, cut to 20.
SCP1_UNITS = ("series_139", "series_219", "series_458")
SCP1_VARIATES = 6
SCP1_LENGTH = 32
RAIN_UNITS = tuple(f"series_0_{number}" for number in range(971, 976))
RAIN_LENGTH = 20
# One record per row, as the collection stores them.
Record = tuple[str, Sequence[float]]
SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("start", pa.string()),
        ("end", pa.string()),
        ("freq", pa.string()),
        ("target", pa.list_(pa.float32())),
    ]
)


def shards(
    root: Path, files: Mapping[str, Sequence[Sequence[Record]]], schema: Any = SCHEMA
) -> Path:
    """Shards of record batches in the collection's layout, one file per name given."""
    root.mkdir(parents=True, exist_ok=True)
    for name, batches in files.items():
        with pa.OSFile(str(root / name), "wb") as sink, ipc.new_stream(sink, schema) as writer:
            for records in batches:
                writer.write_batch(
                    pa.record_batch(
                        [
                            pa.array([item for item, _ in records]),
                            pa.array([""] * len(records)),
                            pa.array([""] * len(records)),
                            pa.array([""] * len(records)),
                            pa.array([list(values) for _, values in records], type=schema[4].type),
                        ],
                        schema=schema,
                    )
                )
    return root


def record(dataset: UtsdDataset, series: int, variate: int, values: Sequence[float]) -> Record:
    return f"{dataset.prefix}_{series}_{variate}", values


@pytest.fixture
def scp1() -> UtsdCorpusReader:
    return UtsdCorpusReader(SAMPLE, SCP1)


@pytest.fixture
def rain() -> UtsdCorpusReader:
    return UtsdCorpusReader(SAMPLE, RAIN)


def test_a_multivariate_dataset_has_a_channel_per_variate(scp1: UtsdCorpusReader) -> None:
    schema = scp1.describe().channel_schema

    assert schema.names == tuple(f"variate_{number:03d}" for number in range(SCP1_VARIATES))
    assert not any(channel.timeless for channel in schema)


def test_a_collection_of_series_has_one_channel(rain: UtsdCorpusReader) -> None:
    assert rain.describe().channel_schema.names == ("value",)


def test_values_at_their_index_are_a_regular_regime(scp1: UtsdCorpusReader) -> None:
    assert scp1.describe().sampling_regime is SamplingRegime.REGULAR


def test_series_are_units_in_the_order_of_their_number_however_the_shards_hold_them(
    scp1: UtsdCorpusReader,
) -> None:
    units = list(scp1.read_units())

    assert [str(unit.key) for unit in units] == list(SCP1_UNITS)
    assert all(unit.extent == TimeExtent(0.0, float(SCP1_LENGTH)) for unit in units)
    assert all(unit.static_features == () for unit in units)
    assert scp1.describe().content.unit_count == len(SCP1_UNITS)
    assert scp1.describe().content.observation_count == (
        len(SCP1_UNITS) * SCP1_VARIATES * SCP1_LENGTH
    )


def test_a_series_whose_variates_lie_in_two_shards_is_read_whole(scp1: UtsdCorpusReader) -> None:
    observations = list(scp1.read_observations(UnitKey("series_139")))

    assert len(observations) == SCP1_VARIATES * SCP1_LENGTH
    assert observations[0] == Observation("variate_000", 0.0, 100.75)
    assert [observation.channel for observation in observations[:SCP1_VARIATES]] == [
        f"variate_{number:03d}" for number in range(SCP1_VARIATES)
    ]
    assert [observation.time for observation in observations[:SCP1_VARIATES]] == [0.0] * 6
    assert observations[-1].time == float(SCP1_LENGTH - 1)


def test_each_series_of_a_collection_is_a_unit_on_the_one_channel(rain: UtsdCorpusReader) -> None:
    units = list(rain.read_units())

    assert [str(unit.key) for unit in units] == list(RAIN_UNITS)
    assert all(unit.extent == TimeExtent(0.0, float(RAIN_LENGTH)) for unit in units)
    assert rain.describe().content.observation_count == len(RAIN_UNITS) * RAIN_LENGTH
    assert {
        observation.channel for observation in rain.read_observations(UnitKey("series_0_973"))
    } == {"value"}


def test_the_checksum_covers_the_dataset_s_records_in_canonical_order(
    scp1: UtsdCorpusReader,
) -> None:
    rows = {}
    for path in sorted(SAMPLE.glob("*.arrow")):
        table = ipc.open_stream(pa.memory_map(str(path))).read_all()
        for item, values in zip(
            table.column("item_id").to_pylist(), table.column("target").to_pylist(), strict=True
        ):
            rows[item] = values
    chunks = []
    for series in (139, 219, 458):
        for variate in range(SCP1_VARIATES):
            name = f"{SCP1.prefix}_{series}_{variate}"
            chunks.append(name.encode() + b"\0")
            chunks.append(np.asarray(rows[name], dtype="<f4").tobytes())

    assert scp1.describe().content.checksum == Checksum.of_chunks(chunks)


def test_two_datasets_of_the_same_shards_describe_differently(
    scp1: UtsdCorpusReader, rain: UtsdCorpusReader
) -> None:
    assert scp1.describe().content.checksum != rain.describe().content.checksum


def test_a_dataset_describes_the_same_however_the_shards_cut_it(tmp_path: Path) -> None:
    records = [record(SCP1, 1, variate, [0.5, 1.5, 2.5]) for variate in range(2)] + [
        record(SCP1, 2, variate, [3.5, 4.5]) for variate in range(2)
    ]
    one = shards(tmp_path / "one", {"data-00000-of-00001.arrow": [records]})
    cut = shards(
        tmp_path / "cut",
        {
            "data-00000-of-00002.arrow": [records[:1], records[1:3]],
            "data-00001-of-00002.arrow": [records[3:]],
        },
    )

    assert UtsdCorpusReader(one, SCP1).describe() == UtsdCorpusReader(cut, SCP1).describe()


def test_a_value_stored_as_not_a_number_is_no_observation(tmp_path: Path) -> None:
    root = shards(
        tmp_path,
        {
            "data.arrow": [
                [record(SCP1, 1, 0, [1.0, math.nan, 3.0]), record(SCP1, 1, 1, [math.nan])]
            ]
        },
    )
    reader = UtsdCorpusReader(root, SCP1)

    (unit,) = reader.read_units()

    assert unit.extent == TimeExtent(0.0, 3.0)
    assert reader.describe().content.observation_count == 2
    assert list(reader.read_observations(unit.key)) == [
        Observation("variate_000", 0.0, 1.0),
        Observation("variate_000", 2.0, 3.0),
    ]


def test_variates_of_unequal_length_walk_index_by_index(tmp_path: Path) -> None:
    root = shards(
        tmp_path, {"data.arrow": [[record(SCP1, 4, 1, [1.0, 2.0]), record(SCP1, 4, 0, [5.0])]]}
    )
    reader = UtsdCorpusReader(root, SCP1)

    (unit,) = reader.read_units()

    assert unit.extent == TimeExtent(0.0, 2.0)
    assert list(reader.read_observations(unit.key)) == [
        Observation("variate_000", 0.0, 5.0),
        Observation("variate_001", 0.0, 1.0),
        Observation("variate_001", 1.0, 2.0),
    ]


def test_records_of_other_datasets_are_passed_over_even_when_their_names_extend_this_one_s(
    tmp_path: Path,
) -> None:
    longer = UtsdDataset(f"{SCP1.prefix}_extended", UtsdLayout.SERIES_OF_VARIATES)
    root = shards(
        tmp_path,
        {
            "data.arrow": [
                [
                    record(SCP1, 1, 0, [1.0]),
                    record(longer, 1, 0, [2.0]),
                    record(RAIN, 0, 7, [3.0]),
                ]
            ]
        },
    )

    assert [str(unit.key) for unit in UtsdCorpusReader(root, SCP1).read_units()] == ["series_1"]
    assert [str(unit.key) for unit in UtsdCorpusReader(root, longer).read_units()] == ["series_1"]


def test_a_collection_names_a_unit_by_both_numbers_however_the_publisher_used_them(
    tmp_path: Path,
) -> None:
    root = shards(
        tmp_path,
        {
            "data.arrow": [
                [
                    record(RAIN, 0, 1, [1.0]),
                    record(RAIN, 1, 0, [2.0]),
                    record(RAIN, 14, 999, [3.0]),
                ]
            ]
        },
    )
    reader = UtsdCorpusReader(root, RAIN)

    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == ["series_0_1", "series_1_0", "series_14_999"]
    assert reader.describe().content.unit_count == 3
    assert list(reader.read_observations(UnitKey("series_14_999"))) == [
        Observation("value", 0.0, 3.0)
    ]


def test_series_that_do_not_all_carry_the_same_variates_contradict_the_multivariate_layout(
    tmp_path: Path,
) -> None:
    root = shards(
        tmp_path,
        {
            "data.arrow": [
                [
                    record(SCP1, 1, 0, [1.0]),
                    record(SCP1, 1, 1, [1.0]),
                    record(SCP1, 2, 0, [2.0]),
                ]
            ]
        },
    )

    with pytest.raises(MalformedCorpusDataError, match="do not all carry the same variates"):
        UtsdCorpusReader(root, SCP1).describe()


def test_a_variate_carried_twice_by_one_series_is_malformed(tmp_path: Path) -> None:
    root = shards(
        tmp_path,
        {
            "data-00000-of-00002.arrow": [[record(SCP1, 1, 0, [1.0])]],
            "data-00001-of-00002.arrow": [[record(SCP1, 1, 0, [2.0])]],
        },
    )

    with pytest.raises(MalformedCorpusDataError, match="variate 0 twice"):
        UtsdCorpusReader(root, SCP1).describe()


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        (f"{SCP1.prefix}_7", "does not name a series and a variate"),
        (f"{SCP1.prefix}_x_1", "does not name a series and a variate"),
    ],
)
def test_a_record_of_the_dataset_without_two_numbers_is_malformed(
    tmp_path: Path, item: str, reason: str
) -> None:
    root = shards(tmp_path, {"data.arrow": [[(item, [1.0])]]})

    with pytest.raises(MalformedCorpusDataError, match=reason) as caught:
        UtsdCorpusReader(root, SCP1).describe()

    assert "data.arrow, row 0" in str(caught.value)


def test_a_record_without_values_is_malformed(tmp_path: Path) -> None:
    root = shards(tmp_path, {"data.arrow": [[record(SCP1, 1, 0, [])]]})

    with pytest.raises(MalformedCorpusDataError, match="holds no value"):
        UtsdCorpusReader(root, SCP1).describe()


def test_a_shard_that_is_not_an_arrow_stream_is_malformed(tmp_path: Path) -> None:
    (tmp_path / "data.arrow").write_bytes(b"not an arrow stream")

    with pytest.raises(MalformedCorpusDataError, match="not an Arrow stream"):
        UtsdCorpusReader(tmp_path, SCP1).describe()


def test_a_shard_of_another_schema_is_malformed(tmp_path: Path) -> None:
    doubles = pa.schema(
        [
            ("item_id", pa.string()),
            ("start", pa.string()),
            ("end", pa.string()),
            ("freq", pa.string()),
            ("target", pa.list_(pa.float64())),
        ]
    )
    root = shards(tmp_path, {"data.arrow": [[record(SCP1, 1, 0, [1.0])]]}, schema=doubles)

    with pytest.raises(MalformedCorpusDataError, match="single-precision"):
        UtsdCorpusReader(root, SCP1).describe()


def test_a_missing_directory_or_one_without_shards_is_missing_data(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="is missing"):
        UtsdCorpusReader(tmp_path / "nowhere", SCP1).describe()
    with pytest.raises(CorpusDataNotFoundError, match="holds no shard"):
        UtsdCorpusReader(tmp_path, SCP1).describe()


def test_a_dataset_no_shard_holds_a_record_of_is_missing_data(tmp_path: Path) -> None:
    root = shards(tmp_path, {"data.arrow": [[record(RAIN, 0, 1, [1.0])]]})

    with pytest.raises(CorpusDataNotFoundError, match="holds no record of"):
        UtsdCorpusReader(root, SCP1).describe()


@pytest.mark.parametrize(
    "key", ["no-such-series", "series_999", "series_139_0", "series_139/0", "139"]
)
def test_a_key_that_could_name_no_series_is_unknown_before_the_stream(
    scp1: UtsdCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        scp1.read_observations(UnitKey(key))


def test_a_declared_dataset_is_found_by_its_corpus_name_and_an_undeclared_one_refused() -> None:
    assert SCP1.corpus_name == "utsd/Health_SelfRegulationSCP1"
    assert SCP1.layout is UtsdLayout.SERIES_OF_VARIATES
    assert RAIN.layout is UtsdLayout.COLLECTION_OF_SERIES
    with pytest.raises(ValueError, match="no dataset of the collection"):
        UtsdCorpusReader.dataset_named("utsd/Nowhere_nothing")
    with pytest.raises(ValueError, match="no dataset of the collection"):
        UtsdCorpusReader.dataset_named("Health_SelfRegulationSCP1")


def test_every_declared_dataset_has_a_distinct_prefix_and_corpus_name() -> None:
    prefixes = [dataset.prefix for dataset in UtsdCorpusReader.DATASETS]
    names = [dataset.corpus_name for dataset in UtsdCorpusReader.DATASETS]

    assert len(set(prefixes)) == len(prefixes)
    assert len(set(names)) == len(names)
    assert all(name.startswith(UtsdCorpusReader.CORPUS_PREFIX) for name in names)


@pytest.mark.skipif(raw_root("utsd") is None, reason="the raw corpus is not on this machine")
def test_every_dataset_the_shards_hold_is_declared_and_describes_as_it_streams() -> None:
    root = raw_root("utsd") or Path()
    held = set()
    for path in sorted(root.glob("*.arrow")):
        reader = ipc.open_stream(pa.memory_map(str(path)))
        for batch in reader:
            for item in batch.column("item_id").to_pylist():
                held.add(item.rsplit("_", 2)[0])

    assert held == {dataset.prefix for dataset in UtsdCorpusReader.DATASETS}
    smallest = UtsdCorpusReader(root, SCP1)
    described = smallest.describe().content
    assert described.observation_count == sum(
        1 for unit in smallest.read_units() for _ in smallest.read_observations(unit.key)
    )
