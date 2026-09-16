import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.smd import SmdCorpusReader
from emblema.catalog.domain.channels.channel_schema import Channel
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
from tests.support.corpora import BUDGET, raw_root, sample

METRICS = SmdCorpusReader.METRICS
SUBSETS = SmdCorpusReader.SUBSETS
WIDTH = len(METRICS)

SAMPLE = sample("smd")
SAMPLE_SUBSETS = ("1", "2")
SAMPLE_ROWS = 60


def row(values: Sequence[str] = ("0.5",) * WIDTH) -> str:
    return ",".join(values)


def render(minutes: int) -> bytes:
    """A well-formed training file of that many minutes."""
    return ("\n".join([row()] * minutes) + "\n").encode("utf-8")


def with_machines(root: Path, files: Mapping[str, bytes]) -> Path:
    for name, content in files.items():
        (root / f"{name}.txt").write_bytes(content)
    return root


def read_one(tmp_path: Path, content: bytes) -> None:
    root = with_machines(tmp_path, {"machine-1-1": content})
    SmdCorpusReader(root, subsets=("1",)).describe()


@pytest.fixture
def reader() -> SmdCorpusReader:
    return SmdCorpusReader(SAMPLE, subsets=SAMPLE_SUBSETS)


def test_channels_are_the_38_metrics_named_after_their_column(reader: SmdCorpusReader) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == WIDTH
    assert Channel("metric_01") in schema.channels
    assert Channel("metric_38") in schema.channels
    # Zero padding makes the name order the column order, so a consumer reading the schema in its
    # own order reads the metrics as the files write them.
    assert schema.names == tuple(metric.name for metric in METRICS)


def test_minute_resolution_is_a_regular_regime(reader: SmdCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.REGULAR


def test_machines_are_units_and_metric_values_are_observations(reader: SmdCorpusReader) -> None:
    content = reader.describe().content

    assert content.unit_count == len(SAMPLE_SUBSETS)
    assert content.observation_count == len(SAMPLE_SUBSETS) * SAMPLE_ROWS * WIDTH


def test_checksum_is_that_of_the_file_bytes(reader: SmdCorpusReader) -> None:
    files = [SAMPLE / "machine-1-1.txt", SAMPLE / "machine-2-1.txt"]

    assert reader.describe().content.checksum == Checksum.of_chunks(
        path.read_bytes() for path in files
    )


def test_groups_are_read_in_canonical_order_however_they_are_named(tmp_path: Path) -> None:
    first, second = render(2), render(3)
    root = with_machines(tmp_path, {"machine-1-1": first, "machine-2-1": second})

    forward = SmdCorpusReader(root, subsets=("1", "2")).describe()
    backward = SmdCorpusReader(root, subsets=("2", "1")).describe()

    assert forward == backward
    assert forward.content.checksum == Checksum.of_chunks([first, second])
    assert forward.content.observation_count == 5 * WIDTH


def test_machines_of_a_group_are_read_in_name_order(tmp_path: Path) -> None:
    files = {"machine-1-10": render(1), "machine-1-2": render(2)}
    root = with_machines(tmp_path, files)

    description = SmdCorpusReader(root, subsets=("1",)).describe()

    assert description.content.checksum == Checksum.of_chunks(
        [files["machine-1-10"], files["machine-1-2"]]
    )


def test_all_three_groups_are_read_by_default(tmp_path: Path) -> None:
    root = with_machines(tmp_path, {f"machine-{group}-1": render(1) for group in SUBSETS})

    assert SmdCorpusReader(root).describe().content.unit_count == 3


def test_the_test_halves_are_not_part_of_the_corpus(tmp_path: Path) -> None:
    root = with_machines(tmp_path, {"machine-1-1": render(1)})
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "machine-1-1.txt").write_bytes(render(5))

    assert SmdCorpusReader(root).describe().content.observation_count == WIDTH


def test_blank_lines_carry_nothing(tmp_path: Path) -> None:
    root = with_machines(tmp_path, {"machine-1-1": render(2).replace(b"\n", b"\n\n")})

    content = SmdCorpusReader(root, subsets=("1",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, 2 * WIDTH)


def test_carriage_returns_belong_to_the_line_ending_not_to_the_last_field(tmp_path: Path) -> None:
    root = with_machines(tmp_path, {"machine-1-1": render(2).replace(b"\n", b"\r\n")})

    content = SmdCorpusReader(root, subsets=("1",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, 2 * WIDTH)


@pytest.mark.parametrize("subsets", [(), ("4",), ("1", "one")])
def test_rejects_unknown_or_no_groups(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="SMD"):
        SmdCorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_directory_is_reported_as_missing_data(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="missing"):
        SmdCorpusReader(tmp_path / "absent").describe()


def test_a_directory_without_machines_is_reported_as_missing_data(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="no machine"):
        SmdCorpusReader(tmp_path).describe()


def test_a_group_without_machines_leaves_the_others_readable(tmp_path: Path) -> None:
    root = with_machines(tmp_path, {"machine-1-1": render(1)})

    assert SmdCorpusReader(root, subsets=("1", "2")).describe().content.unit_count == 1


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "no row"),
        (b"\n\n", "no row"),
        (row(("0.5",) * (WIDTH - 1)).encode(), "columns"),
        (row(("0.5",) * (WIDTH + 1)).encode(), "columns"),
        (row(("x",) + ("0.5",) * (WIDTH - 1)).encode(), "convert"),
        (row(("nan",) + ("0.5",) * (WIDTH - 1)).encode(), "non-finite"),
        (row(("inf",) + ("0.5",) * (WIDTH - 1)).encode(), "non-finite"),
    ],
    ids=["empty", "blank-only", "too-narrow", "too-wide", "text-value", "nan", "inf"],
)
def test_rejects_malformed_files(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason):
        read_one(tmp_path, content)


def test_malformed_data_names_the_file_and_the_line(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match=r"machine-1-1\.txt, line 2"):
        read_one(tmp_path, (row() + "\n0.5").encode())


@given(minutes=st.integers(min_value=1, max_value=8))
def test_counts_and_extents_follow_the_rows_written(
    tmp_path_factory: pytest.TempPathFactory, minutes: int
) -> None:
    root = with_machines(tmp_path_factory.mktemp("smd"), {"machine-2-3": render(minutes)})
    reader = SmdCorpusReader(root, subsets=("2",))

    content = reader.describe().content

    assert content.unit_count == 1
    assert content.observation_count == minutes * WIDTH
    assert [unit.extent for unit in reader.read_units()] == [TimeExtent(0.0, float(minutes))]


def test_machines_are_units_keyed_by_file_spanning_their_minutes(reader: SmdCorpusReader) -> None:
    units = list(reader.read_units())

    assert [(str(unit.key), unit.extent) for unit in units] == [
        ("machine-1-1", TimeExtent(0.0, 60.0)),
        ("machine-2-1", TimeExtent(0.0, 60.0)),
    ]
    assert all(unit.static_features == () for unit in units)


def test_observations_of_a_machine_are_its_metric_values_minute_by_minute(
    reader: SmdCorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey("machine-1-1")))

    assert len(observations) == SAMPLE_ROWS * WIDTH
    assert observations[:3] == [
        Observation("metric_01", 0.0, 0.032258),
        Observation("metric_02", 0.0, 0.039195),
        Observation("metric_03", 0.0, 0.027871),
    ]
    assert observations[-1].time == 59.0


@pytest.mark.parametrize(
    "key", ["machine-3-1", "machine-1-99", "1-1", "machine-1", "machine-1-1-1", "Machine-1-1"]
)
def test_a_key_that_could_name_no_machine_is_unknown_before_the_stream(
    reader: SmdCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_a_machine_from_a_missing_directory_is_missing_data(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="missing"):
        SmdCorpusReader(tmp_path / "absent").read_observations(UnitKey("machine-1-1"))


@pytest.mark.skipif(
    raw_root("smd") is None,
    reason="the raw SMD files are not on this machine (scripts/fetch_corpora.py smd)",
)
def test_full_corpus_agrees_with_the_facts_measured_by_the_data_spike() -> None:
    root = raw_root("smd")
    assert root is not None
    with BUDGET.open("rb") as handle:
        measured = tomllib.load(handle)["corpora"]["smd"]["measured"]

    description = SmdCorpusReader(root).describe()

    assert description.content.unit_count == measured["units"]
    assert len(description.channel_schema) == measured["channels"]
    assert description.content.observation_count == measured["observations"]
