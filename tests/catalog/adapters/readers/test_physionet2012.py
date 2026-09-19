import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.adapters.readers.physionet2012 import Physionet2012CorpusReader
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.corpora import BUDGET, raw_root, sample

SERIES = Physionet2012CorpusReader.SERIES
SUBSETS = Physionet2012CorpusReader.SUBSETS
SAMPLE = sample("physionet2012")
# The four stays of the sample in the order the reader visits them, and how many measurements
# each holds once the descriptors are taken out and a weight recorded as unknown is skipped.
SAMPLE_STAYS = ("set-a/132539", "set-a/132548", "set-a/140501", "set-b/149509")
SAMPLE_OBSERVATIONS = (267, 453, 0, 485)
STAY = TimeExtent(0.0, 48.0 + 1.0 / 60.0)
HEADER = "Time,Parameter,Value"
DESCRIPTORS = (
    "00:00,Age,54",
    "00:00,Gender,0",
    "00:00,Height,170",
    "00:00,ICUType,4",
    "00:00,Weight,80",
)


def stay(*rows: str, record: str = "1", descriptors: Sequence[str] = DESCRIPTORS) -> bytes:
    """A well-formed stay file: the header, the record, its descriptors, then the rows given."""
    lines = [HEADER, f"00:00,RecordID,{record}", *descriptors, *rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def with_stays(root: Path, files: Mapping[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def read_one(tmp_path: Path, content: bytes) -> Physionet2012CorpusReader:
    root = with_stays(tmp_path, {"set-a/1": content})
    reader = Physionet2012CorpusReader(root, subsets=("set-a",))
    reader.describe()
    return reader


@pytest.fixture
def reader() -> Physionet2012CorpusReader:
    return Physionet2012CorpusReader(SAMPLE)


def test_channels_are_the_clinical_variables_and_the_descriptors(
    reader: Physionet2012CorpusReader,
) -> None:
    schema = reader.describe().channel_schema

    timed = [channel for channel in schema if not channel.timeless]
    timeless = [channel.name for channel in schema if channel.timeless]
    assert timed == sorted(SERIES, key=lambda channel: channel.name)
    assert len(timed) == 37
    assert timeless == [
        "Age",
        "Gender",
        "Height",
        "ICUType/cardiac_surgery_recovery",
        "ICUType/coronary_care",
        "ICUType/medical",
        "ICUType/surgical",
    ]


def test_a_clinical_stream_is_an_irregular_regime(reader: Physionet2012CorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_stays_are_units_and_measurements_are_observations(
    reader: Physionet2012CorpusReader,
) -> None:
    content = reader.describe().content

    assert content.unit_count == len(SAMPLE_STAYS)
    assert content.observation_count == sum(SAMPLE_OBSERVATIONS)


def test_checksum_is_that_of_the_file_bytes(reader: Physionet2012CorpusReader) -> None:
    expected = Checksum.of_chunks((SAMPLE / f"{key}.txt").read_bytes() for key in SAMPLE_STAYS)

    assert reader.describe().content.checksum == expected


def test_the_sample_keeps_the_line_endings_the_publisher_wrote() -> None:
    for key in SAMPLE_STAYS:
        assert b"\r" not in (SAMPLE / f"{key}.txt").read_bytes()


def test_every_stay_spans_the_protocols_48_hours(reader: Physionet2012CorpusReader) -> None:
    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == list(SAMPLE_STAYS)
    assert all(unit.extent == STAY for unit in units)


def test_descriptors_become_timeless_features_and_an_unknown_one_none(
    reader: Physionet2012CorpusReader,
) -> None:
    (stay_132539, *_) = reader.read_units()

    # Height and weight are recorded as -1: no feature for the one, no weight for the other.
    assert stay_132539.static_features == (
        StaticFeature("Age", 54.0),
        StaticFeature("Gender", 0.0),
        StaticFeature("ICUType/surgical", 1.0),
    )


def test_a_stay_of_descriptors_alone_is_a_unit_that_measures_nothing(
    reader: Physionet2012CorpusReader,
) -> None:
    assert list(reader.read_observations(UnitKey("set-a/140501"))) == []
    (*_, empty, _) = reader.read_units()
    assert str(empty.key) == "set-a/140501"
    assert len(empty.static_features) == 3


def test_admission_measurements_and_both_weights_are_observations_at_hour_zero(
    reader: Physionet2012CorpusReader,
) -> None:
    observations = list(reader.read_observations(UnitKey("set-b/149509")))

    assert len(observations) == SAMPLE_OBSERVATIONS[-1]
    assert observations[:8] == [
        Observation("DiasABP", 0.0, 83.0),
        Observation("HR", 0.0, 81.0),
        Observation("MAP", 0.0, 104.0),
        Observation("RespRate", 0.0, 20.0),
        Observation("SysABP", 0.0, 140.0),
        Observation("Temp", 0.0, 35.4),
        Observation("Weight", 0.0, 70.8),
        Observation("Weight", 0.0, 70.4),
    ]
    assert observations[8].time == pytest.approx(5.0 / 60.0)


def test_sets_are_read_in_canonical_order_however_they_are_named() -> None:
    reader = Physionet2012CorpusReader(SAMPLE, subsets=("set-b", "set-a"))

    assert [str(unit.key) for unit in reader.read_units()] == list(SAMPLE_STAYS)


def test_both_sets_are_read_by_default(tmp_path: Path) -> None:
    root = with_stays(tmp_path, {"set-a/1": stay(), "set-b/2": stay(record="2")})

    assert Physionet2012CorpusReader(root).describe().content.unit_count == 2


@pytest.mark.parametrize("subsets", [(), ("set-c",), ("set-a", "Set-B")])
def test_rejects_unknown_or_no_sets(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="PhysioNet set"):
        Physionet2012CorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_set_is_reported_as_missing_data(tmp_path: Path) -> None:
    root = with_stays(tmp_path, {"set-a/1": stay()})

    with pytest.raises(CorpusDataNotFoundError, match="set-b"):
        Physionet2012CorpusReader(root).describe()


def test_a_set_without_stays_is_reported_as_missing_data(tmp_path: Path) -> None:
    (tmp_path / "set-a").mkdir()

    with pytest.raises(CorpusDataNotFoundError, match="no stay"):
        Physionet2012CorpusReader(tmp_path, subsets=("set-a",)).describe()


def test_a_descriptor_recorded_as_unknown_is_no_feature_and_no_weight(tmp_path: Path) -> None:
    unknown = ("00:00,Age,54", "00:00,Gender,-1", "00:00,Height,-1", "00:00,ICUType,-1")
    reader = read_one(tmp_path, stay("00:00,Weight,-1", "00:10,HR,80", descriptors=unknown))

    (unit,) = reader.read_units()
    assert unit.static_features == (StaticFeature("Age", 54.0),)
    assert list(reader.read_observations(unit.key)) == [Observation("HR", 1.0 / 6.0, 80.0)]


def test_a_descriptor_the_file_does_not_carry_is_no_feature(tmp_path: Path) -> None:
    reader = read_one(tmp_path, stay("00:10,HR,80", descriptors=("00:00,Age,54",)))

    (unit,) = reader.read_units()
    assert unit.static_features == (StaticFeature("Age", 54.0),)


def test_the_protocols_last_stamp_lies_inside_the_stay(tmp_path: Path) -> None:
    reader = read_one(tmp_path, stay("01:30,HR,80", "48:00,HR,81"))

    observations = list(reader.read_observations(UnitKey("set-a/1")))
    assert [observation.time for observation in observations] == [0.0, 1.5, 48.0]
    assert all(STAY.contains(observation.time) for observation in observations)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "no header"),
        (b"\n\n", "no header"),
        ((HEADER + "\n").encode(), "no RecordID"),
        (b"Time,Value\n", "columns"),
        (b"Time,Param,Value\n", "expected the header"),
        (stay(record="2"), "names record 2, not 1"),
        (stay("00:00,RecordID,1"), "a second RecordID"),
        (stay("03:00,Age,54"), "away from admission"),
        (stay("00:00,Age,55"), "given twice"),
        (stay("00:00,Age,54", descriptors=("00:00,Age,-1",)), "given twice"),
        (stay(descriptors=("00:00,ICUType,5",)), "no ward has the code 5"),
        (stay("00:10,Pulse,80"), "unknown parameter"),
        (stay("0010,HR,80"), "not an HH:MM stamp"),
        # A superscript two is a digit to ``str.isdigit`` and not a number to ``int``.
        (stay("0²:00,HR,80"), "not an HH:MM stamp"),
        (stay("00:60,HR,80"), "not an HH:MM stamp"),
        (stay("48:01,HR,80"), "past the 48-hour protocol"),
        (stay("00:10,HR,80", "00:05,HR,81"), "goes back in time"),
        (stay("00:10,HR,80,1"), "columns"),
        (stay("00:10,HR,x"), "convert"),
        (stay("00:10,HR,nan"), "finite"),
        (stay(descriptors=("00:00,Age,inf",)), "finite"),
    ],
    ids=[
        "empty",
        "blank-only",
        "header-only",
        "too-narrow-header",
        "foreign-header",
        "another-record",
        "second-record-id",
        "descriptor-off-admission",
        "descriptor-twice",
        "descriptor-twice-after-an-unrecorded-one",
        "unknown-ward",
        "unknown-parameter",
        "stamp-without-colon",
        "stamp-outside-ascii",
        "minute-sixty",
        "past-the-protocol",
        "back-in-time",
        "too-wide",
        "text-value",
        "nan",
        "inf-descriptor",
    ],
)
def test_rejects_malformed_files(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason):
        read_one(tmp_path, content)


def test_malformed_data_names_the_file_and_the_line(tmp_path: Path) -> None:
    with pytest.raises(MalformedCorpusDataError, match=r"set-a/1\.txt, line 8"):
        read_one(tmp_path, stay("00:10,Pulse,80"))


@given(minutes=st.lists(st.integers(min_value=0, max_value=48 * 60), min_size=1, max_size=8))
def test_counts_follow_the_rows_written_and_the_extent_does_not(
    tmp_path_factory: pytest.TempPathFactory, minutes: list[int]
) -> None:
    rows = [f"{minute // 60:02d}:{minute % 60:02d},HR,80" for minute in sorted(minutes)]
    root = with_stays(tmp_path_factory.mktemp("physionet2012"), {"set-a/1": stay(*rows)})
    reader = Physionet2012CorpusReader(root, subsets=("set-a",))

    content = reader.describe().content

    assert content.unit_count == 1
    # The rows written, plus the weight taken at admission.
    assert content.observation_count == len(minutes) + 1
    assert [unit.extent for unit in reader.read_units()] == [STAY]


@pytest.mark.parametrize(
    "key",
    ["set-c/132539", "132539", "set-a/999", "set-a/../set-b/149509", "Set-a/132539", "set-a/1/2"],
)
def test_a_key_that_could_name_no_stay_is_unknown_before_the_stream(
    reader: Physionet2012CorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_a_stay_from_a_missing_set_is_missing_data(tmp_path: Path) -> None:
    root = with_stays(tmp_path, {"set-a/1": stay()})

    with pytest.raises(CorpusDataNotFoundError, match="set-b"):
        Physionet2012CorpusReader(root).read_observations(UnitKey("set-b/1"))


# Stays of set A whose weight at admission is recorded. The data spike counted the admission
# weight as a descriptor and left it out; the reader reads it as the first weight of the series.
ADMISSION_WEIGHTS = 3674
# Set B, counted here on 2026-09-19 because the spike counted only its stays. It is the side the
# publication holds out, so nothing else would read it until the corpus is published.
SET_B_STAYS = 4000
SET_B_OBSERVATIONS = 1742188


@pytest.mark.skipif(
    raw_root("physionet2012") is None,
    reason="the raw PhysioNet files are not on this machine (fetch_corpora.py physionet2012)",
)
def test_both_published_sets_agree_with_the_counts_measured_on_them() -> None:
    root = raw_root("physionet2012")
    assert root is not None
    with BUDGET.open("rb") as handle:
        measured = tomllib.load(handle)["corpora"]["physionet2012"]["measured"]

    # Describing a set parses every file the way reading it does, so this is also where a stay of
    # either set that the reader would refuse would surface.
    set_a = Physionet2012CorpusReader(root, subsets=("set-a",)).describe()
    set_b = Physionet2012CorpusReader(root, subsets=("set-b",)).describe()

    assert set_a.content.unit_count == measured["units"]
    assert len(SERIES) == measured["channels"]
    assert set_a.content.observation_count == measured["observations"] + ADMISSION_WEIGHTS
    assert set_b.content.unit_count == SET_B_STAYS
    assert set_b.content.observation_count == SET_B_OBSERVATIONS
    assert set_b.channel_schema == set_a.channel_schema
