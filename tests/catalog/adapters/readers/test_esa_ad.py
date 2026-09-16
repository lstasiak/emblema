import io
import pickle
import shutil
import tomllib
import zipfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
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

LIGHTWEIGHT = EsaAdCorpusReader.LIGHTWEIGHT
SUBSETS = EsaAdCorpusReader.SUBSETS

pd = pytest.importorskip("pandas")
# pandas ships no type information, so a frame is whatever it hands back.
Frame = Any

SAMPLE = sample("esa_ad")
MISSION = "ESA-Mission1"
OTHER = "ESA-Mission2"
# What the thinning left in the training halves: twelve months of Mission1, ten of Mission2, and
# one row per 48 hours in each of the 17 channels.
SAMPLE_UNITS = 22
SAMPLE_OBSERVATIONS = 3105
ORIGIN = datetime(2000, 1, 1)
HOUR = 3600.0


def frame(seconds: Sequence[float], values: Sequence[float] | None = None) -> Frame:
    """A channel as the publisher pickles it: one float column indexed by instants."""
    instants = np.datetime64(ORIGIN, "ns") + (
        np.asarray(seconds, dtype=np.float64) * 1e9
    ).round().astype("timedelta64[ns]")
    column = [0.5] * len(seconds) if values is None else list(values)
    return pd.DataFrame(
        {"channel": np.asarray(column, dtype=np.float32)},
        index=pd.DatetimeIndex(instants, name="datetime"),
    )


def archived(member: Frame, name: str = "channel") -> bytes:
    """The bytes of a zip holding one pickled member, the way a channel ships."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, pickle.dumps(member))
    return buffer.getvalue()


def write_archive(root: Path, mission: str, number: int, content: bytes) -> Path:
    path = root / mission / "channels" / f"channel_{number}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def with_mission(
    root: Path, mission: str, seconds: Sequence[float], values: Sequence[float] | None = None
) -> Path:
    """A mission whose lightweight channels all hold rows at the given seconds."""
    for number in LIGHTWEIGHT[mission]:
        write_archive(root, mission, number, archived(frame(seconds, values)))
    return root


def describe_one(tmp_path: Path, content: bytes) -> None:
    """Describe Mission1 with its first channel replaced by ``content``."""
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])
    write_archive(root, MISSION, 41, content)
    EsaAdCorpusReader(root, subsets=(MISSION,)).describe()


@pytest.fixture
def reader() -> EsaAdCorpusReader:
    return EsaAdCorpusReader(SAMPLE)


def test_channels_are_the_lightweight_ones_named_by_their_mission(
    reader: EsaAdCorpusReader,
) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == 17
    assert Channel("ESA-Mission1/channel_41") in schema.channels
    assert Channel("ESA-Mission2/channel_28") in schema.channels
    assert all(channel.unit is None and not channel.timeless for channel in schema)


def test_satellite_telemetry_is_an_irregular_regime(reader: EsaAdCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_months_are_units_and_kept_instants_are_observations(reader: EsaAdCorpusReader) -> None:
    content = reader.describe().content

    assert content.unit_count == SAMPLE_UNITS
    assert content.observation_count == SAMPLE_OBSERVATIONS


def test_checksum_is_that_of_the_archive_bytes_in_mission_and_channel_order(
    reader: EsaAdCorpusReader,
) -> None:
    archives = [
        SAMPLE / mission / "channels" / f"channel_{number}.zip"
        for mission in SUBSETS
        for number in LIGHTWEIGHT[mission]
    ]

    assert reader.describe().content.checksum == Checksum.of_chunks(
        path.read_bytes() for path in archives
    )


def test_missions_are_read_in_canonical_order_however_they_are_named(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, 4 * HOUR])
    with_mission(tmp_path, OTHER, [0.0, HOUR, 4 * HOUR])

    forward = EsaAdCorpusReader(tmp_path, subsets=(MISSION, OTHER)).describe()
    backward = EsaAdCorpusReader(tmp_path, subsets=(OTHER, MISSION)).describe()

    assert forward == backward
    assert forward.content.checksum == Checksum.of_chunks(
        (tmp_path / mission / "channels" / f"channel_{number}.zip").read_bytes()
        for mission in SUBSETS
        for number in LIGHTWEIGHT[mission]
    )
    assert forward.content.observation_count == 6 + 2 * 11


def test_both_missions_are_read_by_default(reader: EsaAdCorpusReader) -> None:
    missions = {str(unit.key).partition("/")[0] for unit in reader.read_units()}

    assert missions == set(SUBSETS)


@pytest.mark.parametrize("subsets", [(), ("ESA-Mission3",), ("ESA-Mission1", "mission2")])
def test_rejects_unknown_or_no_missions(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="ESA mission"):
        EsaAdCorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_mission_is_reported_as_missing_data(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, HOUR])

    with pytest.raises(CorpusDataNotFoundError, match="ESA-Mission2"):
        EsaAdCorpusReader(tmp_path).describe()


def test_a_missing_channel_archive_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])
    (root / MISSION / "channels" / "channel_44.zip").unlink()

    with pytest.raises(CorpusDataNotFoundError, match="channel_44"):
        EsaAdCorpusReader(root, subsets=(MISSION,)).describe()


def two_members() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("channel", pickle.dumps(frame([0.0])))
        archive.writestr("another", pickle.dumps(frame([0.0])))
    return buffer.getvalue()


def raw_member(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("channel", content)
    return buffer.getvalue()


def with_plain_index() -> Frame:
    plain = frame([0.0, HOUR])
    return plain.reset_index(drop=True)


def with_missing_instant() -> Frame:
    broken = frame([0.0, HOUR])
    return broken.set_axis(pd.DatetimeIndex([broken.index[0], pd.NaT]))


def with_two_columns() -> Frame:
    wide = frame([0.0, HOUR])
    wide["second"] = wide["channel"]
    return wide


def with_text_values() -> Frame:
    return frame([0.0, HOUR]).astype({"channel": str})


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"not a zip", "not a zip"),
        (two_members(), "one member"),
        (raw_member(b"junk"), "not a pickled data frame"),
        (archived([1.0, 2.0]), "one column"),
        (archived(with_two_columns()), "one column"),
        (archived(with_plain_index()), "instants as the index"),
        (archived(with_missing_instant()), "instant is missing"),
        (archived(frame([HOUR, 0.0])), "goes back in time"),
        (archived(with_text_values()), "not numbers"),
        (archived(frame([0.0, HOUR], [0.5, float("nan")])), "non-finite"),
        (archived(frame([0.0, HOUR], [float("inf"), 0.5])), "non-finite"),
    ],
    ids=[
        "not-a-zip",
        "two-members",
        "not-a-pickle",
        "not-a-frame",
        "two-columns",
        "plain-index",
        "missing-instant",
        "backwards",
        "text-values",
        "nan",
        "inf",
    ],
)
def test_rejects_malformed_archives(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason):
        describe_one(tmp_path, content)


def test_malformed_data_names_the_archive(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match=r"channel_41\.zip"):
        describe_one(tmp_path, b"not a zip")


def test_a_mission_without_observations_is_malformed(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [])

    with pytest.raises(MalformedCorpusDataError, match="no observation"):
        EsaAdCorpusReader(root, subsets=(MISSION,)).describe()


def test_a_mission_shorter_than_two_bins_has_no_training_half(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, 40.0])

    with pytest.raises(MalformedCorpusDataError, match="training half"):
        EsaAdCorpusReader(root, subsets=(MISSION,)).describe()


def test_the_training_half_ends_at_the_bin_boundary_before_the_exact_half(
    tmp_path: Path,
) -> None:
    # The span is 300 s, the exact half 150 s, the last bin boundary before it 150 s itself: the
    # rows at 0 and 100 s are kept, the ones at 200 and 300 s are the test set.
    root = with_mission(tmp_path, MISSION, [0.0, 100.0, 200.0, 300.0])
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    assert reader.describe().content.observation_count == 2 * 6
    assert [unit.extent for unit in reader.read_units()] == [TimeExtent(0.0, 150.0 / HOUR)]


def test_a_bin_keeps_its_earliest_observation(tmp_path: Path) -> None:
    root = with_mission(
        tmp_path, MISSION, [0.0, 10.0, 20.0, 35.0, 1000.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    kept = [
        (observation.time, observation.value)
        for observation in reader.read_observations(UnitKey(f"{MISSION}/2000-01"))
        if observation.channel == f"{MISSION}/channel_41"
    ]

    assert kept == [(0.0, 1.0), (35.0 / HOUR, 4.0)]


def test_a_channel_without_rows_stays_silent(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR, 4 * HOUR])
    write_archive(root, MISSION, 42, archived(frame([])))
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    description = reader.describe()

    assert len(description.channel_schema) == 6
    assert description.content.observation_count == 2 * 5


def test_a_channel_observed_only_in_the_second_half_stays_silent(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR, 4 * HOUR])
    write_archive(root, MISSION, 42, archived(frame([3 * HOUR, 4 * HOUR])))
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    channels = {o.channel for o in reader.read_observations(UnitKey(f"{MISSION}/2000-01"))}

    assert f"{MISSION}/channel_42" not in channels
    assert len(channels) == 5


def test_units_are_the_calendar_months_of_the_training_half(tmp_path: Path) -> None:
    # A mission from 15 January to 15 May: its training half runs to noon on 15 March, so the
    # months are a partial January, February whole, and March up to the cut.
    start = timedelta(days=14).total_seconds()
    root = with_mission(tmp_path, MISSION, [start, start + timedelta(days=121).total_seconds()])
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    units = list(reader.read_units())

    assert [(str(unit.key), unit.extent) for unit in units] == [
        (f"{MISSION}/2000-01", TimeExtent(0.0, 17 * 24.0)),
        (f"{MISSION}/2000-02", TimeExtent(17 * 24.0, (17 + 29) * 24.0)),
        (f"{MISSION}/2000-03", TimeExtent((17 + 29) * 24.0, (17 + 29 + 14.5) * 24.0)),
    ]
    assert all(unit.static_features == () for unit in units)


def test_a_month_without_telemetry_is_a_unit_that_yields_nothing(tmp_path: Path) -> None:
    start = timedelta(days=14).total_seconds()
    root = with_mission(tmp_path, MISSION, [start, start + timedelta(days=121).total_seconds()])
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    assert list(reader.read_observations(UnitKey(f"{MISSION}/2000-02"))) == []
    assert len(list(reader.read_observations(UnitKey(f"{MISSION}/2000-01")))) == 6


def test_observations_of_a_month_are_merged_across_channels_in_time_order(
    tmp_path: Path,
) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, 100 * HOUR], [1.0, 9.0])
    write_archive(root, MISSION, 41, archived(frame([0.0, HOUR], [1.0, 2.0])))
    write_archive(root, MISSION, 42, archived(frame([HOUR / 2, HOUR], [3.0, 4.0])))
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))

    observations = list(reader.read_observations(UnitKey(f"{MISSION}/2000-01")))

    assert observations == [
        Observation(f"{MISSION}/channel_41", 0.0, 1.0),
        *(Observation(f"{MISSION}/channel_{number}", 0.0, 1.0) for number in range(43, 47)),
        Observation(f"{MISSION}/channel_42", 0.5, 3.0),
        Observation(f"{MISSION}/channel_41", 1.0, 2.0),
        Observation(f"{MISSION}/channel_42", 1.0, 4.0),
    ]


def test_instants_keep_their_milliseconds_in_hours_since_the_mission_start(
    reader: EsaAdCorpusReader,
) -> None:
    first = next(iter(reader.read_observations(UnitKey(f"{MISSION}/2000-01"))))

    # The mission's first instant is 16.353 s past midnight; its axis starts at the whole second.
    assert first == Observation(f"{MISSION}/channel_41", 353e6 / 3.6e12, 0.8125782012939453)


def test_sample_months_span_from_the_first_instant_to_the_calendar_boundaries(
    reader: EsaAdCorpusReader,
) -> None:
    units = list(reader.read_units())

    assert [(str(unit.key), unit.extent) for unit in units[:2]] == [
        (f"{MISSION}/2000-01", TimeExtent(0.0, (31 * 86400 - 16) / HOUR)),
        (f"{MISSION}/2000-02", TimeExtent((31 * 86400 - 16) / HOUR, (60 * 86400 - 16) / HOUR)),
    ]
    assert str(units[12].key) == f"{OTHER}/2000-01"
    assert units[12].extent.start == 0.0


@pytest.mark.parametrize(
    "key",
    [
        "ESA-Mission3/2000-01",
        "2000-01",
        "ESA-Mission1",
        "ESA-Mission1/2010-01",
        "ESA-Mission1/2000-1",
        "ESA-Mission1/2000-01/x",
        "esa-mission1/2000-01",
    ],
)
def test_a_key_that_could_name_no_month_is_unknown_before_the_stream(
    reader: EsaAdCorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_a_month_of_a_missing_mission_is_missing_data(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])

    with pytest.raises(CorpusDataNotFoundError, match="ESA-Mission2"):
        EsaAdCorpusReader(root).read_observations(UnitKey(f"{OTHER}/2000-01"))


def test_the_mission_last_read_is_served_without_its_archives(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, 2 * HOUR])
    with_mission(tmp_path, OTHER, [0.0, 2 * HOUR])
    reader = EsaAdCorpusReader(tmp_path)
    first = list(reader.read_observations(UnitKey(f"{MISSION}/2000-01")))
    shutil.rmtree(tmp_path / MISSION)

    assert list(reader.read_observations(UnitKey(f"{MISSION}/2000-01"))) == first

    list(reader.read_observations(UnitKey(f"{OTHER}/2000-01")))
    with pytest.raises(CorpusDataNotFoundError):
        reader.read_observations(UnitKey(f"{MISSION}/2000-01"))


def months_between(first_second: int, last_second: int) -> int:
    first = ORIGIN + timedelta(seconds=first_second)
    last = ORIGIN + timedelta(seconds=last_second)
    return (last.year - first.year) * 12 + last.month - first.month + 1


@given(
    seconds=st.lists(st.integers(min_value=0, max_value=20_000_000), min_size=1, max_size=12).map(
        sorted
    )
)
def test_counts_follow_the_rows_written(
    tmp_path_factory: pytest.TempPathFactory, seconds: list[int]
) -> None:
    root = with_mission(tmp_path_factory.mktemp("esa_ad"), MISSION, seconds)
    reader = EsaAdCorpusReader(root, subsets=(MISSION,))
    # The rule, stated independently: the half of the whole-second span, cut back to a bin
    # boundary; one observation per occupied bin before it.
    first, last = seconds[0], seconds[-1]
    end = (first + (last - first) // 2) // 30 * 30

    if end <= first:
        with pytest.raises(MalformedCorpusDataError):
            reader.describe()
        return
    content = reader.describe().content

    assert content.observation_count == 6 * len(
        {second // 30 for second in seconds if second < end}
    )
    assert content.unit_count == months_between(first, end - 1)
    assert next(unit.extent.start for unit in reader.read_units()) == 0.0


@pytest.mark.skipif(
    raw_root("esa_ad") is None,
    reason="the raw ESA-AD files are not on this machine (scripts/fetch_corpora.py esa_ad)",
)
def test_full_corpus_agrees_with_the_facts_measured_by_the_data_spike() -> None:
    root = raw_root("esa_ad")
    assert root is not None
    with BUDGET.open("rb") as handle:
        measured = tomllib.load(handle)["corpora"]["esa_ad"]["measured"]
    reader = EsaAdCorpusReader(root)

    description = reader.describe()

    assert len(description.channel_schema) == measured["channels"]
    assert description.content.observation_count == measured["observations"]
    # The spike counts missions, the independent units; the reader's units are their months.
    missions = {str(unit.key).partition("/")[0] for unit in reader.read_units()}
    assert len(missions) == measured["units"]
