import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.opssat_ad import OpssatAdCorpusReader
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

SAMPLE = sample("opssat_ad")
# What the sample holds: five fragments of the authors' training part, two of their test part,
# twelve rows each but the shortest fragment of the file, which has eight.
SAMPLE_UNITS = {"train/1", "train/6", "train/219", "train/333", "train/1069", "test/5", "test/13"}
SAMPLE_ROWS = 80
HEADER = "channel,timestamp,value,label,sampling,anomaly,segment,train"


def row(
    second: int,
    *,
    channel: str = "CADC0872",
    value: str = "0.5",
    sampling: str = "1",
    anomaly: str = "0",
    segment: str = "1",
    train: str = "1",
) -> str:
    label = "anomaly" if anomaly == "1" else "a3"
    return (
        f"{channel},2022-06-01T23:42:{second:02d}.000Z,{value},{label},{sampling},{anomaly},"
        f"{segment},{train}"
    )


def render(rows: Sequence[str], header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


def with_file(root: Path, content: bytes) -> Path:
    (root / "segments.csv").write_bytes(content)
    return root


def read(tmp_path: Path, content: bytes) -> OpssatAdCorpusReader:
    return OpssatAdCorpusReader(with_file(tmp_path, content))


@pytest.fixture
def reader() -> OpssatAdCorpusReader:
    return OpssatAdCorpusReader(SAMPLE)


def test_channels_are_the_nine_the_satellite_reports(reader: OpssatAdCorpusReader) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == 9
    assert Channel("CADC0872") in schema.channels
    assert all(channel.unit is None for channel in schema.channels)


def test_the_regime_promises_no_cadence(reader: OpssatAdCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_the_description_counts_the_fragments_and_their_rows(
    reader: OpssatAdCorpusReader,
) -> None:
    content = reader.describe().content

    assert content.unit_count == len(SAMPLE_UNITS)
    assert content.observation_count == SAMPLE_ROWS
    assert content.checksum == Checksum.of_bytes((SAMPLE / "segments.csv").read_bytes())


def test_units_are_the_fragments_keyed_by_the_authors_part(
    reader: OpssatAdCorpusReader,
) -> None:
    units = {str(unit.key) for unit in reader.read_units()}

    assert units == SAMPLE_UNITS


def test_a_fragment_spans_from_its_first_row_to_one_period_past_its_last(
    reader: OpssatAdCorpusReader,
) -> None:
    units = {str(unit.key): unit for unit in reader.read_units()}

    # Twelve rows a second apart, then the period of one second.
    assert units["train/1"].extent == TimeExtent(0.0, 12.0)
    # Eight rows five seconds apart, then the period of five.
    assert units["train/1069"].extent == TimeExtent(0.0, 40.0)


def test_observations_are_one_channel_in_seconds_since_the_first_row(
    reader: OpssatAdCorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey("train/1")))

    assert len(observations) == 12
    assert observations[0] == Observation("CADC0872", 0.0, -2.1487e-05)
    assert observations[1].time == 1.0
    assert {observation.channel for observation in observations} == {"CADC0872"}


def test_only_the_selected_part_is_read(reader: OpssatAdCorpusReader) -> None:
    tested = OpssatAdCorpusReader(SAMPLE, subsets=("test",))

    assert {str(unit.key) for unit in tested.read_units()} == {"test/5", "test/13"}
    assert tested.describe().content.unit_count == 2
    # The checksum is the file's, whichever part is read: it is the same data.
    assert tested.describe().content.checksum == reader.describe().content.checksum
    with pytest.raises(UnknownUnitError, match="selected part"):
        tested.read_observations(UnitKey("train/1"))


def test_the_counts_agree_with_the_streams(reader: OpssatAdCorpusReader) -> None:
    content = reader.describe().content
    units = list(reader.read_units())

    assert len(units) == content.unit_count
    assert (
        sum(len(list(reader.read_observations(u.key))) for u in units) == content.observation_count
    )


@pytest.mark.parametrize("key", ["train/999", "valve1/1", "1"])
def test_a_fragment_the_file_does_not_hold_is_refused(
    reader: OpssatAdCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="missing"):
        OpssatAdCorpusReader(tmp_path).describe()


def test_an_unknown_part_is_refused_up_front() -> None:
    with pytest.raises(ValueError, match="OPSSAT-AD part"):
        OpssatAdCorpusReader(SAMPLE, subsets=("validation",))


def test_another_header_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match="columns"):
        read(tmp_path, render([row(0)], header="segment,channel,value")).describe()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([row(0, channel="CADC9999")], "unknown channel"),
        ([row(0, value="nan")], "non-finite"),
        ([row(0, value="hot")], "could not convert"),
        ([row(0, sampling="0")], "positive"),
        ([row(0, anomaly="2")], "0 or 1"),
        ([row(0, segment="0")], "positive number"),
        ([row(0), row(1, channel="CADC0873")], "share one channel"),
        ([row(0), row(1, sampling="5")], "share one channel"),
        ([row(0), row(1, train="0")], "share one channel"),
        ([row(5), row(4)], "back in time"),
        (["CADC0872,2022-06-01T23:42:54.000Z,0.5,a3,1,0,1"], "columns"),
    ],
    ids=[
        "channel",
        "non-finite value",
        "value not a number",
        "sampling",
        "anomaly flag",
        "segment number",
        "channel changes",
        "sampling changes",
        "part changes",
        "time goes back",
        "width",
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, rows: Sequence[str], message: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=message):
        read(tmp_path, render(rows)).describe()


def test_the_full_corpus_matches_the_benchmark_as_published() -> None:
    root = raw_root("opssat_ad")
    if root is None:
        pytest.skip("OPSSAT-AD is not downloaded")
    estimate = tomllib.loads(BUDGET.read_text(encoding="utf-8"))["corpora"]["opssat_ad"]["estimate"]

    reader = OpssatAdCorpusReader(root)
    content = reader.describe().content
    units = list(reader.read_units())

    # The benchmark publishes 2,123 fragments, 1,594 of them in the authors' training part, which
    # is what the budget table estimates; the corpus is never an ingredient, so it is not measured.
    assert content.unit_count == len(units) == 2123
    assert sum(1 for unit in units if unit.key.belongs_to("train")) == estimate["units"] == 1594
    assert content.observation_count == 303_493
