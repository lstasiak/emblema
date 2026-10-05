from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from emblema.catalog.adapters.readers.tep import TennesseeEastmanCorpusReader
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

rdata = pytest.importorskip("rdata")
pd = pytest.importorskip("pandas")
# pandas ships no type information, so a frame is whatever it hands back.
Frame = Any

COLUMNS = TennesseeEastmanCorpusReader.COLUMNS
SUBSETS = TennesseeEastmanCorpusReader.SUBSETS
FAULT_FREE, FAULTY = SUBSETS
SAMPLE = sample("tep")
# What the sample holds: two runs per file, each cut to its first twenty samples.
SAMPLE_RUNS = (
    f"{FAULT_FREE}/fault00-run001",
    f"{FAULT_FREE}/fault00-run002",
    f"{FAULTY}/fault01-run001",
    f"{FAULTY}/fault02-run001",
)
SAMPLE_LENGTH = 20
VARIABLES = len(COLUMNS) - 3
# The variable names the publisher uses for the objects of the two files.
OBJECTS = {FAULT_FREE: "fault_free_training", FAULTY: "faulty_training"}


def frame(runs: Sequence[tuple[int, int, int]], value: float = 0.5) -> Frame:
    """A data frame in the publisher's layout: ``(fault, run, samples)`` per run, one value."""
    rows = [
        [float(fault), float(run), sample, *([value] * VARIABLES)]
        for fault, run, samples in runs
        for sample in range(1, samples + 1)
    ]
    built = pd.DataFrame(rows, columns=list(COLUMNS))
    built["sample"] = built["sample"].astype("Int32")
    return built


def written(root: Path, files: Mapping[str, Frame], names: Mapping[str, str] = OBJECTS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for subset, content in files.items():
        rdata.write_rda(root / f"{subset}.RData", {names[subset]: content})
    return root


@pytest.fixture
def reader() -> TennesseeEastmanCorpusReader:
    return TennesseeEastmanCorpusReader(SAMPLE)


def test_channels_are_the_process_variables_and_nothing_else(
    reader: TennesseeEastmanCorpusReader,
) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == VARIABLES
    assert schema.names == tuple(sorted(COLUMNS[3:]))
    assert not any(channel.timeless for channel in schema)
    assert {"faultNumber", "simulationRun", "sample"}.isdisjoint(schema.names)


def test_a_simulation_sampled_every_three_minutes_is_a_regular_regime(
    reader: TennesseeEastmanCorpusReader,
) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.REGULAR


def test_runs_are_units_and_every_variable_of_every_sample_an_observation(
    reader: TennesseeEastmanCorpusReader,
) -> None:
    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == list(SAMPLE_RUNS)
    assert all(unit.extent == TimeExtent(1.0, SAMPLE_LENGTH + 1.0) for unit in units)
    assert all(unit.static_features == () for unit in units)
    assert reader.describe().content.unit_count == len(SAMPLE_RUNS)
    assert reader.describe().content.observation_count == (
        len(SAMPLE_RUNS) * SAMPLE_LENGTH * VARIABLES
    )


def test_observations_walk_the_samples_in_order_with_the_variables_in_column_order(
    reader: TennesseeEastmanCorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey(SAMPLE_RUNS[2])))

    assert [observation.time for observation in observations[:VARIABLES]] == [1.0] * VARIABLES
    assert [observation.channel for observation in observations[:VARIABLES]] == list(COLUMNS[3:])
    assert observations[-1].time == float(SAMPLE_LENGTH)
    assert len(observations) == SAMPLE_LENGTH * VARIABLES


def test_checksum_is_that_of_the_file_bytes_in_subset_order(
    reader: TennesseeEastmanCorpusReader,
) -> None:
    expected = Checksum.of_bytes(
        b"".join((SAMPLE / f"{subset}.RData").read_bytes() for subset in SUBSETS)
    )

    assert reader.describe().content.checksum == expected


def test_subsets_are_read_in_canonical_order_however_they_are_named() -> None:
    reversed_reader = TennesseeEastmanCorpusReader(SAMPLE, subsets=(FAULTY, FAULT_FREE))

    assert reversed_reader.describe() == TennesseeEastmanCorpusReader(SAMPLE).describe()
    assert [str(unit.key) for unit in reversed_reader.read_units()] == list(SAMPLE_RUNS)


def test_one_file_alone_is_a_corpus_of_its_runs() -> None:
    faulty = TennesseeEastmanCorpusReader(SAMPLE, subsets=(FAULTY,))

    assert [str(unit.key) for unit in faulty.read_units()] == list(SAMPLE_RUNS[2:])
    assert faulty.describe().content.unit_count == 2


@pytest.mark.parametrize("subsets", [(), ("TEP_Faulty_Testing",), (FAULT_FREE, "faulty")])
def test_rejects_unknown_or_no_files(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="Tennessee Eastman file"):
        TennesseeEastmanCorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_file_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = written(tmp_path, {FAULT_FREE: frame([(0, 1, 3)])})

    with pytest.raises(CorpusDataNotFoundError, match=r"TEP_Faulty_Training\.RData is missing"):
        TennesseeEastmanCorpusReader(root).describe()


def test_the_object_may_carry_any_name_as_long_as_it_is_alone(tmp_path: Path) -> None:
    root = written(
        tmp_path,
        {FAULT_FREE: frame([(0, 1, 3)])},
        names={FAULT_FREE: "whatever_the_publisher_called_it"},
    )

    units = list(TennesseeEastmanCorpusReader(root, subsets=(FAULT_FREE,)).read_units())

    assert [str(unit.key) for unit in units] == [f"{FAULT_FREE}/fault00-run001"]


def test_runs_are_ordered_by_fault_then_run_however_the_rows_lie(tmp_path: Path) -> None:
    shuffled = frame([(2, 1, 2), (1, 2, 2), (1, 1, 2)]).sample(frac=1.0, random_state=3)
    root = written(tmp_path, {FAULTY: shuffled})
    reader = TennesseeEastmanCorpusReader(root, subsets=(FAULTY,))

    assert [unit.key.name for unit in reader.read_units()] == [
        "fault01-run001",
        "fault01-run002",
        "fault02-run001",
    ]
    assert [o.time for o in reader.read_observations(UnitKey(f"{FAULTY}/fault02-run001"))] == [
        *([1.0] * VARIABLES),
        *([2.0] * VARIABLES),
    ]


def test_a_run_of_one_sample_spans_one_unit_of_time(tmp_path: Path) -> None:
    root = written(tmp_path, {FAULT_FREE: frame([(0, 7, 1)])})

    (unit,) = TennesseeEastmanCorpusReader(root, subsets=(FAULT_FREE,)).read_units()

    assert (unit.key.name, unit.extent) == ("fault00-run007", TimeExtent(1.0, 2.0))


def test_a_value_is_read_as_the_file_carries_it(tmp_path: Path) -> None:
    root = written(tmp_path, {FAULT_FREE: frame([(0, 1, 1)], value=3674.0)})

    (first, *_) = TennesseeEastmanCorpusReader(root, subsets=(FAULT_FREE,)).read_observations(
        UnitKey(f"{FAULT_FREE}/fault00-run001")
    )

    assert first == Observation("xmeas_1", 1.0, 3674.0)


def _two_objects(root: Path) -> None:
    rdata.write_rda(
        root / f"{FAULT_FREE}.RData",
        {"fault_free_training": frame([(0, 1, 2)]), "another": frame([(0, 2, 2)])},
    )


def _not_a_frame(root: Path) -> None:
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": [1.0, 2.0, 3.0]})


def _wrong_columns(root: Path) -> None:
    built = frame([(0, 1, 2)]).rename(columns={"xmv_11": "xmv_12"})
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": built})


def _non_finite(root: Path) -> None:
    built = frame([(0, 1, 2)])
    built.loc[1, "xmeas_3"] = np.nan
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": built})


def _fractional_fault(root: Path) -> None:
    built = frame([(0, 1, 2)])
    built["faultNumber"] = 0.5
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": built})


def _sample_gap(root: Path) -> None:
    built = frame([(0, 1, 3)])
    built = built[built["sample"] != 2]
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": built})


def _samples_from_zero(root: Path) -> None:
    built = frame([(0, 1, 2)])
    built["sample"] = built["sample"] - 1
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": built})


def _beyond_the_widths(root: Path) -> None:
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": frame([(100, 1, 2)])})


def _no_rows(root: Path) -> None:
    rdata.write_rda(root / f"{FAULT_FREE}.RData", {"fault_free_training": frame([])})


def _not_r_data(root: Path) -> None:
    (root / f"{FAULT_FREE}.RData").write_bytes(b"not an R data file at all")


@pytest.mark.parametrize(
    ("spoil", "reason"),
    [
        (_two_objects, "one object"),
        (_not_a_frame, "data frame"),
        (_wrong_columns, "columns"),
        (_non_finite, "non-finite"),
        (_fractional_fault, "faultNumber is not a whole number"),
        (_sample_gap, "numbered 1..L"),
        (_samples_from_zero, "numbered 1..L"),
        (_beyond_the_widths, "beyond the publisher's counts"),
        (_no_rows, "no rows"),
        (_not_r_data, "not an R data file"),
    ],
    ids=[
        "two_objects",
        "not_a_frame",
        "wrong_columns",
        "non_finite",
        "fractional_fault",
        "sample_gap",
        "samples_from_zero",
        "beyond_the_widths",
        "no_rows",
        "not_r_data",
    ],
)
def test_rejects_malformed_files(tmp_path: Path, spoil: Any, reason: str) -> None:
    spoil(tmp_path)

    with pytest.raises(MalformedCorpusDataError, match=reason) as caught:
        TennesseeEastmanCorpusReader(tmp_path, subsets=(FAULT_FREE,)).describe()

    assert f"{FAULT_FREE}.RData" in str(caught.value)


@pytest.mark.parametrize(
    "key",
    [
        "no-such-run",
        f"{FAULT_FREE}/fault00-run999",
        f"{FAULTY}/fault00-run001",
        f"{FAULT_FREE}/fault0-run1",
        f"{FAULT_FREE}/../fault00-run001",
        "TEP_Faulty_Testing/fault01-run001",
    ],
)
def test_a_key_that_could_name_no_run_is_unknown_before_the_stream(
    reader: TennesseeEastmanCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_a_run_from_a_missing_file_is_missing_data(tmp_path: Path) -> None:
    root = written(tmp_path, {FAULT_FREE: frame([(0, 1, 3)])})

    with pytest.raises(CorpusDataNotFoundError):
        TennesseeEastmanCorpusReader(root).read_observations(UnitKey(f"{FAULTY}/fault01-run001"))


@pytest.mark.skipif(raw_root("tep") is None, reason="the raw corpus is not on this machine")
def test_both_published_files_describe_as_many_observations_as_they_stream() -> None:
    reader = TennesseeEastmanCorpusReader(raw_root("tep") or Path())

    described = reader.describe().content
    units = list(reader.read_units())

    assert described.unit_count == len(units) == 21 * 500
    assert all(unit.extent == TimeExtent(1.0, 501.0) for unit in units)
    assert described.observation_count == sum(
        1 for unit in units for _ in reader.read_observations(unit.key)
    )
