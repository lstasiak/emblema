import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.skab import SENSORS, SUBSETS, SkabCorpusReader
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

SAMPLE = sample("skab")
SAMPLE_SUBSETS = ("anomaly-free", "other", "valve1")
SAMPLE_ROWS = 60
HEADER = "datetime;" + ";".join(sensor.name for sensor in SENSORS)
LABELLED_HEADER = HEADER + ";anomaly;changepoint"


def row(second: int, values: Sequence[str] = ("0.5",) * 8, labels: str = "") -> str:
    return f"2020-03-01 15:44:{second:02d};" + ";".join(values) + labels


def render(seconds: Sequence[int], *, labelled: bool = True) -> bytes:
    """A well-formed experiment file with a row at each of the given seconds."""
    header = LABELLED_HEADER if labelled else HEADER
    labels = ";0.0;0.0" if labelled else ""
    lines = [header, *(row(second, labels=labels) for second in seconds)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def with_experiments(root: Path, files: Mapping[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def read_one(tmp_path: Path, content: bytes) -> None:
    root = with_experiments(tmp_path, {"valve1/0": content})
    SkabCorpusReader(root, subsets=("valve1",)).describe()


@pytest.fixture
def reader() -> SkabCorpusReader:
    return SkabCorpusReader(SAMPLE, subsets=SAMPLE_SUBSETS)


def test_channels_are_the_eight_sensors_named_by_the_header(reader: SkabCorpusReader) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == 8
    assert Channel("Current", "A") in schema.channels
    assert Channel("Volume Flow RateRMS", "L/min") in schema.channels
    assert schema.names == tuple(sorted(sensor.name for sensor in SENSORS))


def test_the_testbed_cadence_is_a_regular_regime(reader: SkabCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.REGULAR


def test_experiments_are_units_and_sensor_values_are_observations(
    reader: SkabCorpusReader,
) -> None:
    content = reader.describe().content

    assert content.unit_count == len(SAMPLE_SUBSETS)
    assert content.observation_count == len(SAMPLE_SUBSETS) * SAMPLE_ROWS * len(SENSORS)


def test_checksum_is_that_of_the_file_bytes(reader: SkabCorpusReader) -> None:
    files = [
        SAMPLE / "anomaly-free" / "anomaly-free.csv",
        SAMPLE / "other" / "1.csv",
        SAMPLE / "valve1" / "0.csv",
    ]

    assert reader.describe().content.checksum == Checksum.of_chunks(
        path.read_bytes() for path in files
    )


def test_the_sample_keeps_the_line_endings_the_publisher_wrote() -> None:
    # A normalising checkout or editor would rewrite these and silently change every checksum the
    # reader computes over the sample, so the two kinds have to survive side by side in the
    # repository. `.gitattributes` exempts this directory; this is what would notice if it stopped.
    assert b"\r\n" in (SAMPLE / "valve1" / "0.csv").read_bytes()
    assert b"\r\n" not in (SAMPLE / "other" / "1.csv").read_bytes()


def test_folders_are_read_in_canonical_order_however_they_are_named(tmp_path: Path) -> None:
    first, second = render([1, 2]), render([1, 2, 3])
    root = with_experiments(tmp_path, {"other/1": first, "valve1/0": second})

    forward = SkabCorpusReader(root, subsets=("other", "valve1")).describe()
    backward = SkabCorpusReader(root, subsets=("valve1", "other")).describe()

    assert forward == backward
    assert forward.content.checksum == Checksum.of_chunks([first, second])
    assert forward.content.observation_count == 5 * len(SENSORS)


def test_files_of_a_folder_are_read_in_name_order(tmp_path: Path) -> None:
    files = {"other/10": render([1]), "other/2": render([1, 2])}
    root = with_experiments(tmp_path, files)

    description = SkabCorpusReader(root, subsets=("other",)).describe()

    assert description.content.checksum == Checksum.of_chunks([files["other/10"], files["other/2"]])


def test_all_four_folders_are_read_by_default(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {f"{name}/0": render([1]) for name in SUBSETS})

    assert SkabCorpusReader(root).describe().content.unit_count == 4


def test_the_anomaly_free_recording_carries_no_label_columns(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {"anomaly-free/anomaly-free": render([1], labelled=False)})

    content = SkabCorpusReader(root, subsets=("anomaly-free",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, len(SENSORS))


def test_blank_lines_carry_nothing(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {"valve1/0": render([1, 2]).replace(b"\n", b"\n\n")})

    content = SkabCorpusReader(root, subsets=("valve1",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, 2 * len(SENSORS))


def test_carriage_returns_belong_to_the_line_ending_not_to_the_last_field(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {"valve1/0": render([1, 2]).replace(b"\n", b"\r\n")})

    content = SkabCorpusReader(root, subsets=("valve1",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, 2 * len(SENSORS))


@pytest.mark.parametrize("subsets", [(), ("valve3",), ("valve1", "Other")])
def test_rejects_unknown_or_no_folders(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="SKAB"):
        SkabCorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_folder_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {"valve1/0": render([1])})

    with pytest.raises(CorpusDataNotFoundError, match="valve2"):
        SkabCorpusReader(root, subsets=("valve1", "valve2")).describe()


def test_a_folder_without_experiments_is_reported_as_missing_data(tmp_path: Path) -> None:
    (tmp_path / "valve2").mkdir()

    with pytest.raises(CorpusDataNotFoundError, match="no experiment"):
        SkabCorpusReader(tmp_path, subsets=("valve2",)).describe()


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "no header"),
        (b"\n\n", "no header"),
        ((LABELLED_HEADER + "\n").encode(), "no row"),
        (b"datetime;Current\n", "expected the columns"),
        ((HEADER + ";anomaly\n").encode(), "expected no columns"),
        ((LABELLED_HEADER + "\n" + row(1, ("0.5",) * 7, ";0.0;0.0")).encode(), "columns"),
        ((LABELLED_HEADER + "\n" + row(1, ("x",) + ("0.5",) * 7, ";0.0;0.0")).encode(), "convert"),
        ((LABELLED_HEADER + "\n" + "later;" + ";".join(("0.5",) * 10)).encode(), "isoformat"),
        ((LABELLED_HEADER + "\n" + row(1, ("nan",) + ("0.5",) * 7, ";0.0;0.0")).encode(), "finite"),
        ((LABELLED_HEADER + "\n" + row(1, ("inf",) + ("0.5",) * 7, ";0.0;0.0")).encode(), "finite"),
    ],
    ids=[
        "empty",
        "blank-only",
        "header-only",
        "foreign-header",
        "one-label-column",
        "too-narrow",
        "text-value",
        "unreadable-timestamp",
        "nan",
        "inf",
    ],
)
def test_rejects_malformed_files(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason):
        read_one(tmp_path, content)


def test_malformed_data_names_the_file_and_the_line(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match=r"0\.csv, line 3"):
        read_one(tmp_path, (LABELLED_HEADER + "\n" + row(1, labels=";0.0;0.0") + "\n1;2").encode())


def test_a_row_going_back_in_time_is_malformed(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match="goes back in time"):
        read_one(tmp_path, render([5, 3]))


@given(seconds=st.lists(st.integers(min_value=0, max_value=59), min_size=1, max_size=8).map(sorted))
def test_counts_and_extents_follow_the_rows_written(
    tmp_path_factory: pytest.TempPathFactory, seconds: list[int]
) -> None:
    root = with_experiments(tmp_path_factory.mktemp("skab"), {"valve1/0": render(seconds)})
    reader = SkabCorpusReader(root, subsets=("valve1",))

    content = reader.describe().content

    assert content.unit_count == 1
    assert content.observation_count == len(seconds) * len(SENSORS)
    # The extent runs from the first row to one second past the last, whatever lies between.
    assert [unit.extent for unit in reader.read_units()] == [
        TimeExtent(0.0, float(seconds[-1] - seconds[0] + 1))
    ]


def test_experiments_are_units_keyed_by_folder_and_file_spanning_their_rows(
    reader: SkabCorpusReader,
) -> None:
    units = list(reader.read_units())

    assert [(str(unit.key), unit.extent) for unit in units] == [
        ("anomaly-free/anomaly-free", TimeExtent(0.0, 64.0)),
        ("other/1", TimeExtent(0.0, 63.0)),
        ("valve1/0", TimeExtent(0.0, 62.0)),
    ]
    assert all(unit.static_features == () for unit in units)


def test_observations_of_an_experiment_are_its_sensor_values_second_by_second(
    reader: SkabCorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey("other/1")))

    assert len(observations) == SAMPLE_ROWS * len(SENSORS)
    assert observations[:3] == [
        Observation("Accelerometer1RMS", 0.0, 0.0820647),
        Observation("Accelerometer2RMS", 0.0, 0.133521),
        Observation("Current", 0.0, 1.27794),
    ]
    assert observations[-1].time == 62.0


@pytest.mark.parametrize(
    "key", ["valve2/0", "1", "other/99", "other/../valve1/0", "Other/1", "other/1/2"]
)
def test_a_key_that_could_name_no_experiment_is_unknown_before_the_stream(
    reader: SkabCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_an_experiment_from_a_missing_folder_is_missing_data(tmp_path: Path) -> None:
    root = with_experiments(tmp_path, {"valve1/0": render([1])})

    with pytest.raises(CorpusDataNotFoundError, match="valve2"):
        SkabCorpusReader(root, subsets=("valve1", "valve2")).read_observations(UnitKey("valve2/0"))


@pytest.mark.skipif(
    raw_root("skab") is None,
    reason="the raw SKAB files are not on this machine (scripts/fetch_corpora.py skab)",
)
def test_full_corpus_agrees_with_the_facts_measured_by_the_data_spike() -> None:
    root = raw_root("skab")
    assert root is not None
    with BUDGET.open("rb") as handle:
        measured = tomllib.load(handle)["corpora"]["skab"]["measured"]

    description = SkabCorpusReader(root).describe()

    assert description.content.unit_count == measured["units"]
    assert len(description.channel_schema) == measured["channels"]
    assert description.content.observation_count == measured["observations"]
