import io
import pickle
import zipfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from emblema.catalog.adapters.readers.esa_ad_raw import (
    ORIGIN_SECONDS,
    EsaAdRawCorpusReader,
    EsaAdRawPolicy,
)
from emblema.catalog.domain.channels.channel_schema import Channel
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.corpora import raw_root, sample

POLICY = EsaAdRawCorpusReader.POLICY
SUBSETS = EsaAdRawCorpusReader.SUBSETS

pd = pytest.importorskip("pandas")
# pandas ships no type information, so a frame is whatever it hands back.
Frame = Any

SAMPLE = sample("esa_ad")
MISSION = "ESA-Mission1"
OTHER = "ESA-Mission2"
ORIGIN = datetime(2000, 1, 1)
HOUR = 3600.0
# A policy of a few channels per mission, so a test writes a few archives rather than 59.
SMALL = EsaAdRawPolicy(channels={MISSION: (41, 42), OTHER: (18,)}, cap=1_000_000)


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
    root: Path,
    mission: str,
    seconds: Sequence[float],
    values: Sequence[float] | None = None,
    policy: EsaAdRawPolicy = SMALL,
) -> Path:
    """A mission whose selected channels all hold rows at the given seconds."""
    for number in policy.channels[mission]:
        write_archive(root, mission, number, archived(frame(seconds, values)))
    return root


def reader_of(root: Path, *missions: str, policy: EsaAdRawPolicy = SMALL) -> EsaAdRawCorpusReader:
    return EsaAdRawCorpusReader(root, subsets=missions or SUBSETS, policy=policy)


@pytest.fixture
def reader() -> EsaAdRawCorpusReader:
    return EsaAdRawCorpusReader(SAMPLE)


def test_the_policy_in_force_reads_the_target_channels_of_three_packets_and_a_group() -> None:
    assert POLICY.channels[MISSION] == (*range(12, 53), *range(70, 77))
    assert POLICY.channels[OTHER] == tuple(range(18, 29))
    assert len(POLICY.channels[MISSION]) == 48


def test_channels_are_the_policys_named_by_their_mission(reader: EsaAdRawCorpusReader) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == 59
    assert Channel("ESA-Mission1/channel_12") in schema.channels
    assert Channel("ESA-Mission1/channel_76") in schema.channels
    assert Channel("ESA-Mission2/channel_28") in schema.channels
    assert all(channel.unit is None and not channel.timeless for channel in schema)


def test_satellite_telemetry_is_an_irregular_regime(reader: EsaAdRawCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_the_checksum_opens_with_the_policy_and_covers_the_archives_in_order(
    reader: EsaAdRawCorpusReader,
) -> None:
    archives = [
        SAMPLE / mission / "channels" / f"channel_{number}.zip"
        for mission in SUBSETS
        for number in POLICY.channels[mission]
    ]

    assert reader.describe().content.checksum == Checksum.of_chunks(
        [POLICY.statement(), *(path.read_bytes() for path in archives)]
    )


def test_another_selection_of_the_same_files_is_another_checksum(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, HOUR])
    narrower = EsaAdRawPolicy(channels={MISSION: (41,)}, cap=SMALL.cap)

    both = reader_of(tmp_path, MISSION).describe().content.checksum
    one = reader_of(tmp_path, MISSION, policy=narrower).describe().content.checksum

    assert both != one


def test_the_whole_span_is_read_and_the_months_are_its_units(tmp_path: Path) -> None:
    # A mission from 15 January to 15 May, read whole: five months, the first and last partial.
    start = timedelta(days=14).total_seconds()
    end = start + timedelta(days=121).total_seconds()
    root = with_mission(tmp_path, MISSION, [start, end])
    reader = reader_of(root, MISSION)

    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == [
        f"{MISSION}/2000-0{month}" for month in range(1, 6)
    ]
    assert units[0].extent == TimeExtent(14 * 24.0, 31 * 24.0)
    assert units[-1].extent == TimeExtent((31 + 29 + 31 + 30) * 24.0, 135 * 24.0 + 1 / HOUR)
    assert reader.describe().content.observation_count == 2 * 2


def test_instants_lie_in_hours_since_the_shared_origin(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [10 * HOUR, 10 * HOUR + 30.0])
    reader = reader_of(root, MISSION)

    observations = list(reader.read_observations(UnitKey(f"{MISSION}/2000-01")))

    assert [observation.time for observation in observations] == pytest.approx(
        [10.0, 10.0, 10.0 + 30.0 / HOUR, 10.0 + 30.0 / HOUR]
    )
    assert datetime.fromtimestamp(ORIGIN_SECONDS, tz=None).year >= 1999


def test_every_reading_is_kept_at_its_own_instant_below_the_cap(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, 0.4, 0.8, 30.0], [1.0, 2.0, 3.0, 4.0])

    kept = [
        (observation.time * HOUR, observation.value)
        for observation in reader_of(root, MISSION).read_observations(UnitKey(f"{MISSION}/2000-01"))
        if observation.channel == f"{MISSION}/channel_41"
    ]

    assert kept == pytest.approx([(0.0, 1.0), (0.4, 2.0), (0.8, 3.0), (30.0, 4.0)])


def test_a_channel_over_the_cap_keeps_the_earliest_reading_of_the_smallest_bin_under_it(
    tmp_path: Path,
) -> None:
    # Six readings, a cap of three: one-second bins leave four, two-second bins leave three.
    seconds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    root = with_mission(tmp_path, MISSION, seconds, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    capped = EsaAdRawPolicy(channels=SMALL.channels, cap=3)

    kept = [
        (observation.time * HOUR, observation.value)
        for observation in reader_of(root, MISSION, policy=capped).read_observations(
            UnitKey(f"{MISSION}/2000-01")
        )
        if observation.channel == f"{MISSION}/channel_41"
    ]

    assert kept == pytest.approx([(0.0, 1.0), (2.0, 5.0)])
    assert reader_of(root, MISSION, policy=capped).describe().content.observation_count == 2 * 2


def test_missions_are_read_in_canonical_order_however_they_are_named(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, 4 * HOUR])
    with_mission(tmp_path, OTHER, [0.0, HOUR, 4 * HOUR])

    forward = reader_of(tmp_path, MISSION, OTHER).describe()
    backward = reader_of(tmp_path, OTHER, MISSION).describe()

    assert forward == backward
    assert forward.content.observation_count == 2 * 2 + 3


def test_a_mission_the_policy_selects_nothing_of_is_refused_up_front(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selects no channel"):
        reader_of(tmp_path, OTHER, policy=EsaAdRawPolicy(channels={MISSION: (41,)}, cap=10))


@pytest.mark.parametrize("subsets", [(), ("ESA-Mission3",), ("ESA-Mission1", "mission2")])
def test_rejects_unknown_or_no_missions(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="ESA mission"):
        EsaAdRawCorpusReader(SAMPLE, subsets=subsets)


@pytest.mark.parametrize(
    ("channels", "cap", "reason"),
    [
        ({MISSION: (42, 41)}, 10, "distinct and in order"),
        ({MISSION: (41, 41)}, 10, "distinct and in order"),
        ({MISSION: ()}, 10, "distinct and in order"),
        ({MISSION: (41,)}, 0, "at least one reading"),
    ],
    ids=["unordered", "repeated", "empty", "cap"],
)
def test_a_policy_that_contradicts_itself_is_refused(
    channels: dict[str, tuple[int, ...]], cap: int, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        EsaAdRawPolicy(channels=channels, cap=cap)


def test_a_missing_channel_archive_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])
    (root / MISSION / "channels" / "channel_42.zip").unlink()

    with pytest.raises(CorpusDataNotFoundError, match="channel_42"):
        reader_of(root, MISSION).describe()


def test_a_missing_mission_is_reported_as_missing_data(tmp_path: Path) -> None:
    with_mission(tmp_path, MISSION, [0.0, HOUR])

    with pytest.raises(CorpusDataNotFoundError, match="ESA-Mission2"):
        reader_of(tmp_path).describe()


def test_an_instant_before_the_origin_is_malformed(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])
    write_archive(root, MISSION, 41, archived(frame([-HOUR, HOUR])))

    with pytest.raises(MalformedCorpusDataError, match="precedes"):
        reader_of(root, MISSION).describe()


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"not a zip", "not a zip"),
        (archived([1.0, 2.0]), "one column"),
        (archived(frame([HOUR, 0.0])), "goes back in time"),
        (archived(frame([0.0, HOUR], [0.5, float("nan")])), "non-finite"),
    ],
    ids=["not-a-zip", "not-a-frame", "backwards", "nan"],
)
def test_rejects_malformed_archives(tmp_path: Path, content: bytes, reason: str) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])
    write_archive(root, MISSION, 41, content)

    with pytest.raises(MalformedCorpusDataError, match=reason):
        reader_of(root, MISSION).describe()


def test_a_mission_without_observations_is_malformed(tmp_path: Path) -> None:
    root = with_mission(tmp_path, MISSION, [])

    with pytest.raises(MalformedCorpusDataError, match="no observation"):
        reader_of(root, MISSION).describe()


@pytest.mark.parametrize("key", ["ESA-Mission1/1999-12", "ESA-Mission2/2000-01", "2000-01"])
def test_a_month_outside_the_span_or_the_selection_is_refused(tmp_path: Path, key: str) -> None:
    root = with_mission(tmp_path, MISSION, [0.0, HOUR])

    with pytest.raises(UnknownUnitError):
        reader_of(root, MISSION).read_observations(UnitKey(key))


def test_the_counts_agree_with_the_streams(reader: EsaAdRawCorpusReader) -> None:
    content = reader.describe().content
    units = list(reader.read_units())

    assert len(units) == content.unit_count
    assert (
        sum(len(list(reader.read_observations(u.key))) for u in units) == content.observation_count
    )


def test_the_full_corpus_keeps_every_reading_of_the_policys_channels() -> None:
    root = raw_root("esa_ad")
    if root is None:
        pytest.skip("the ESA Anomaly Dataset is not downloaded")
    reader = EsaAdRawCorpusReader(root, subsets=(OTHER,))

    content = reader.describe().content
    units = list(reader.read_units())

    # Mission2's lightweight channels hold 7,387,262 readings each, none thinned, over the
    # 42 months from January 2000 to June 2003.
    assert content.observation_count == 11 * 7_387_262
    assert content.unit_count == len(units) == 42
