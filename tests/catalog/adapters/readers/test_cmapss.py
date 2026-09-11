import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.cmapss import SENSORS, SUBSETS, CmapssCorpusReader
from emblema.catalog.domain.channel_schema import Channel
from emblema.catalog.domain.exceptions import CorpusDataNotFoundError, MalformedCorpusDataError
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE = REPO_ROOT / "tests" / "data" / "cmapss"
SAMPLE_BYTES = (SAMPLE / "train_FD001.txt").read_bytes()
SAMPLE_ENGINES = 2
SAMPLE_ROWS = 263
RAW = REPO_ROOT / "data" / "raw" / "cmapss"
BUDGET = REPO_ROOT / "scripts" / "corpus_budget.toml"


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

    content = CmapssCorpusReader(root, subsets=("FD002",)).describe().content

    assert content.unit_count == len(cycles)
    assert content.observation_count == sum(cycles) * len(SENSORS)


def raw_root() -> Path | None:
    hits = sorted(RAW.rglob("train_FD001.txt")) if RAW.is_dir() else []
    return hits[0].parent if hits else None


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
