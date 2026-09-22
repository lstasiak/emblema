import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
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
from tests.support.corpora import BUDGET, SAMPLE, raw_root

SENSORS = CmapssCorpusReader.SENSORS
SUBSETS = CmapssCorpusReader.SUBSETS

SAMPLE_BYTES = (SAMPLE / "train_FD001.txt").read_bytes()
SAMPLE_ENGINES = 2
SAMPLE_ROWS = 263


def row(unit: int, cycle: int, values: Sequence[str] = ("0.5",) * 24) -> str:
    return f"{unit} {cycle} " + " ".join(values) + "  "


def render(cycles_per_engine: Sequence[int]) -> bytes:
    """A well-formed training file with the given number of cycles per engine."""
    lines = [
        row(unit, cycle)
        for unit, cycles in enumerate(cycles_per_engine, 1)
        for cycle in range(1, cycles + 1)
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def with_subsets(root: Path, files: Mapping[str, bytes]) -> Path:
    root.mkdir(exist_ok=True)
    for name, content in files.items():
        (root / f"train_{name}.txt").write_bytes(content)
    return root


def read_fd001(tmp_path: Path, content: bytes) -> None:
    CmapssCorpusReader(with_subsets(tmp_path, {"FD001": content}), subsets=("FD001",)).describe()


@pytest.fixture
def reader() -> CmapssCorpusReader:
    return CmapssCorpusReader(SAMPLE, subsets=("FD001",))


def test_channels_are_the_21_sensors_named_after_the_paper(reader: CmapssCorpusReader) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == 21
    assert Channel("T2", "°R") in schema.channels
    assert Channel("epr") in schema.channels
    assert schema.names == tuple(sorted(sensor.name for sensor in SENSORS))


def test_cycles_are_a_regular_regime(reader: CmapssCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.REGULAR


def test_engines_are_units_and_sensor_values_are_observations(reader: CmapssCorpusReader) -> None:
    content = reader.describe().content

    assert content.unit_count == SAMPLE_ENGINES
    assert content.observation_count == SAMPLE_ROWS * len(SENSORS)


def test_checksum_is_that_of_the_file_bytes(reader: CmapssCorpusReader) -> None:
    assert reader.describe().content.checksum == Checksum.of_bytes(SAMPLE_BYTES)


def test_subsets_are_read_in_canonical_order_however_they_are_named(tmp_path: Path) -> None:
    other = render([3, 4])
    root = with_subsets(tmp_path, {"FD001": SAMPLE_BYTES, "FD003": other})

    forward = CmapssCorpusReader(root, subsets=("FD001", "FD003")).describe()
    backward = CmapssCorpusReader(root, subsets=("FD003", "FD001")).describe()

    assert forward == backward
    assert forward.content.checksum == Checksum.of_chunks([SAMPLE_BYTES, other])
    assert forward.content.unit_count == SAMPLE_ENGINES + 2
    assert forward.content.observation_count == (SAMPLE_ROWS + 7) * len(SENSORS)


def test_all_four_subsets_are_read_by_default(tmp_path: Path) -> None:
    root = with_subsets(tmp_path, dict.fromkeys(SUBSETS, render([1])))

    assert CmapssCorpusReader(root).describe().content.unit_count == 4


def test_blank_lines_carry_nothing(tmp_path: Path) -> None:
    padded = render([2]).replace(b"\n", b"\n\n")
    root = with_subsets(tmp_path, {"FD001": padded})

    content = CmapssCorpusReader(root, subsets=("FD001",)).describe().content

    assert (content.unit_count, content.observation_count) == (1, 2 * len(SENSORS))


@pytest.mark.parametrize("subsets", [(), ("FD005",), ("FD001", "fd002")])
def test_rejects_unknown_or_no_subsets(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="subset"):
        CmapssCorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_file_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = with_subsets(tmp_path, {"FD001": render([1])})

    with pytest.raises(CorpusDataNotFoundError, match=r"train_FD002\.txt"):
        CmapssCorpusReader(root, subsets=("FD001", "FD002")).describe()


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "no engine"),
        (b"\n\n", "no engine"),
        (row(1, 1, ("0.5",) * 23).encode(), "columns"),
        (row(1, 1, ("0.5",) * 25).encode(), "columns"),
        (row(1, 1, ("x",) + ("0.5",) * 23).encode(), "could not convert"),
        (b"1.0 1 " + b" ".join([b"0.5"] * 24), "invalid literal"),
        (row(1, 1, ("nan",) + ("0.5",) * 23).encode(), "non-finite"),
        (row(1, 1, ("inf",) + ("0.5",) * 23).encode(), "non-finite"),
        ((row(1, 1) + "\n" + row(1, 3)).encode(), "jumps to cycle 3 after cycle 1"),
        (row(1, 2).encode(), "jumps to cycle 2 after cycle 0"),
    ],
    ids=[
        "empty",
        "blank-only",
        "too-narrow",
        "too-wide",
        "text-value",
        "fractional-unit",
        "nan",
        "inf",
        "cycle-gap",
        "first-cycle-not-one",
    ],
)
def test_rejects_malformed_rows(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason):
        read_fd001(tmp_path, content)


def test_malformed_data_names_the_file_and_the_line(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match=r"train_FD001\.txt, line 2"):
        read_fd001(tmp_path, (row(1, 1) + "\n" + row(1, 5)).encode())


@given(cycles=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=5))
def test_counts_follow_the_engines_written(
    tmp_path_factory: pytest.TempPathFactory, cycles: list[int]
) -> None:
    root = with_subsets(tmp_path_factory.mktemp("cmapss"), {"FD002": render(cycles)})
    reader = CmapssCorpusReader(root, subsets=("FD002",))

    content = reader.describe().content

    assert content.unit_count == len(cycles)
    assert content.observation_count == sum(cycles) * len(SENSORS)
    assert [unit.extent for unit in reader.read_units()] == [
        TimeExtent(1.0, length + 1.0) for length in cycles
    ]


def test_engines_are_units_keyed_by_subset_and_number_spanning_their_cycles(
    reader: CmapssCorpusReader,
) -> None:
    units = list(reader.read_units())

    assert [(str(unit.key), unit.extent) for unit in units] == [
        ("FD001/39", TimeExtent(1.0, 129.0)),
        ("FD001/91", TimeExtent(1.0, 136.0)),
    ]
    assert all(unit.static_features == () for unit in units)


def test_observations_of_an_engine_are_its_sensor_values_cycle_by_cycle(
    reader: CmapssCorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey("FD001/39")))

    assert len(observations) == 128 * len(SENSORS)
    assert observations[:3] == [
        Observation("T2", 1.0, 518.67),
        Observation("T24", 1.0, 642.72),
        Observation("T30", 1.0, 1592.37),
    ]
    assert observations[-1].time == 128.0


@pytest.mark.parametrize("key", ["FD002/39", "39", "FD001/x", "FD001/39/1", "fd001/39"])
def test_a_key_that_could_name_no_engine_is_unknown_before_the_stream(
    reader: CmapssCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_an_engine_the_file_does_not_hold_is_unknown_by_the_end_of_the_stream(
    reader: CmapssCorpusReader,
) -> None:
    with pytest.raises(UnknownUnitError):
        list(reader.read_observations(UnitKey("FD001/7")))


def test_an_engine_whose_rows_are_interrupted_is_malformed(tmp_path: Path) -> None:
    interrupted = "\n".join([row(1, 1), row(2, 1), row(1, 2)]).encode()

    with pytest.raises(MalformedCorpusDataError, match="engine 1 resumes after engine 2"):
        read_fd001(tmp_path, interrupted)


def test_reading_an_engine_with_a_cycle_gap_is_malformed(tmp_path: Path) -> None:
    root = with_subsets(tmp_path, {"FD001": "\n".join([row(1, 1), row(1, 3)]).encode()})

    with pytest.raises(MalformedCorpusDataError, match="jumps to cycle 3"):
        list(CmapssCorpusReader(root, subsets=("FD001",)).read_observations(UnitKey("FD001/1")))


def test_reading_an_engine_from_a_missing_file_is_missing_data(tmp_path: Path) -> None:
    root = with_subsets(tmp_path, {"FD001": render([1])})

    with pytest.raises(CorpusDataNotFoundError, match=r"train_FD002\.txt"):
        CmapssCorpusReader(root, subsets=("FD001", "FD002")).read_observations(UnitKey("FD002/1"))


@pytest.mark.skipif(
    raw_root() is None,
    reason="the raw C-MAPSS files are not on this machine (scripts/fetch_corpora.py cmapss)",
)
def test_full_corpus_agrees_with_the_facts_measured_by_the_data_spike() -> None:
    root = raw_root()
    assert root is not None
    with BUDGET.open("rb") as handle:
        measured = tomllib.load(handle)["corpora"]["cmapss"]["measured"]

    description = CmapssCorpusReader(root).describe()

    assert description.content.unit_count == measured["units"]
    assert len(description.channel_schema) == measured["channels"]
    assert description.content.observation_count == measured["observations"]


SEA_LEVEL = "0kft-M0.00-TRA100"
# Settings as the released files scatter them around four of the six conditions.
FLOWN = {
    SEA_LEVEL: ("-0.0007", "0.0003", "100.0"),
    "42kft-M0.84-TRA100": ("42.0049", "0.8405", "100.0"),
    "25kft-M0.62-TRA60": ("24.9988", "0.6218", "60.0"),
    "10kft-M0.25-TRA100": ("9.9981", "0.2500", "100.0"),
}


def flown_row(unit: int, cycle: int, settings: Sequence[str], sensor: str = "1.5") -> str:
    return row(unit, cycle, (*settings, *(sensor,) * len(SENSORS)))


def per_condition(root: Path, subsets: tuple[str, ...]) -> CmapssCorpusReader:
    return CmapssCorpusReader(root, subsets=subsets, per_condition=True)


def test_per_condition_a_single_condition_subset_has_the_sensors_of_the_first_condition() -> None:
    schema = per_condition(SAMPLE, ("FD001",)).describe().channel_schema

    assert schema.channels == frozenset(
        Channel(f"{sensor.name}@{SEA_LEVEL}", sensor.unit) for sensor in SENSORS
    )


def test_per_condition_a_subset_flown_at_six_conditions_has_a_channel_per_sensor_and_each(
    tmp_path: Path,
) -> None:
    # The schema is the subset's, not the rows': it names all six even where the file flies four.
    content = "\n".join(
        flown_row(1, cycle, settings) for cycle, settings in enumerate(FLOWN.values(), 1)
    )
    root = with_subsets(tmp_path, {"FD002": (content + "\n").encode("ascii")})

    schema = per_condition(root, ("FD002",)).describe().channel_schema

    assert len(schema) == 6 * len(SENSORS)
    assert Channel("T24@42kft-M0.84-TRA100", "°R") in schema.channels
    assert Channel("T24@35kft-M0.84-TRA100", "°R") in schema.channels


def test_per_condition_each_cycle_lands_on_the_channels_of_the_condition_it_was_flown_at(
    tmp_path: Path,
) -> None:
    labels = list(FLOWN)
    content = "\n".join(
        flown_row(1, cycle, FLOWN[label], sensor=f"{cycle}.5")
        for cycle, label in enumerate(labels, 1)
    )
    root = with_subsets(tmp_path, {"FD004": (content + "\n").encode("ascii")})

    observations = list(per_condition(root, ("FD004",)).read_observations(UnitKey("FD004/1")))

    assert len(observations) == len(labels) * len(SENSORS)
    for cycle, label in enumerate(labels, 1):
        at_cycle = [o for o in observations if o.time == float(cycle)]
        assert [o.channel for o in at_cycle] == [f"{sensor.name}@{label}" for sensor in SENSORS]
        assert {o.value for o in at_cycle} == {cycle + 0.5}


def test_per_condition_the_content_and_the_readings_are_the_default_ones_renamed(
    reader: CmapssCorpusReader,
) -> None:
    qualified = per_condition(SAMPLE, ("FD001",))

    assert qualified.describe().content == reader.describe().content
    assert list(qualified.read_units()) == list(reader.read_units())
    plain = list(reader.read_observations(UnitKey("FD001/39")))
    renamed = list(qualified.read_observations(UnitKey("FD001/39")))
    assert [(o.time, o.value) for o in renamed] == [(o.time, o.value) for o in plain]
    assert [o.channel for o in renamed] == [f"{o.channel}@{SEA_LEVEL}" for o in plain]


@pytest.mark.parametrize(
    ("subset", "settings"),
    [
        ("FD002", ("30.0", "0.50", "80.0")),
        ("FD002", ("42.0", "0.62", "100.0")),
        ("FD002", ("25.0", "0.62", "100.0")),
        ("FD001", ("42.0049", "0.8405", "100.0")),
    ],
    ids=["no condition", "altitude of one, Mach of another", "throttle off", "not flown there"],
)
def test_per_condition_a_row_whose_settings_name_none_of_its_subsets_conditions_is_malformed(
    tmp_path: Path, subset: str, settings: tuple[str, str, str]
) -> None:
    content = flown_row(1, 1, FLOWN[SEA_LEVEL]) + "\n" + flown_row(1, 2, settings) + "\n"
    root = with_subsets(tmp_path, {subset: content.encode("ascii")})
    reader = per_condition(root, (subset,))

    with pytest.raises(MalformedCorpusDataError, match=f"train_{subset}.txt, line 2"):
        reader.describe()
    with pytest.raises(MalformedCorpusDataError, match="name none of the conditions"):
        list(reader.read_observations(UnitKey(f"{subset}/1")))


def test_by_default_the_settings_are_left_out_whatever_they_hold(tmp_path: Path) -> None:
    content = flown_row(1, 1, ("30.0", "0.50", "80.0")) + "\n"
    root = with_subsets(tmp_path, {"FD002": content.encode("ascii")})

    observations = list(CmapssCorpusReader(root, ("FD002",)).read_observations(UnitKey("FD002/1")))

    assert [o.channel for o in observations] == [sensor.name for sensor in SENSORS]


@pytest.mark.skipif(
    raw_root() is None,
    reason="the raw C-MAPSS files are not on this machine (scripts/fetch_corpora.py cmapss)",
)
def test_full_corpus_per_condition_names_a_condition_for_every_row() -> None:
    root = raw_root()
    assert root is not None

    description = per_condition(root, SUBSETS).describe()

    assert len(description.channel_schema) == 6 * len(SENSORS)
    assert description.content == CmapssCorpusReader(root).describe().content
