from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.physionet2019 import Physionet2019CorpusReader
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
from tests.support.corpora import raw_root, sample

SERIES = Physionet2019CorpusReader.SERIES
SUBSETS = Physionet2019CorpusReader.SUBSETS
HEADER = "|".join(Physionet2019CorpusReader.HEADER)
SAMPLE = sample("physionet2019")
# The four stays of the sample in the order the reader visits them, how many hours each spans and
# how many measurements each holds once the empty cells are left out.
SAMPLE_STAYS = (
    "training_setA/p000001",
    "training_setA/p000015",
    "training_setA/p000022",
    "training_setB/p100006",
)
SAMPLE_EXTENTS = (
    TimeExtent(1.0, 55.0),
    TimeExtent(1.0, 16.0),
    TimeExtent(5.0, 24.0),
    TimeExtent(6.0, 53.0),
)
SAMPLE_OBSERVATIONS = (294, 128, 183, 219)
TIMED = len(SERIES)
# Age, gender, the two ward flags and the admission offset, in the publisher's column order.
DESCRIPTORS = ("54", "0", "NaN", "NaN", "-0.03")


def row(
    hour: float,
    measured: Mapping[str, str] | None = None,
    descriptors: Sequence[str] = DESCRIPTORS,
    label: str = "0",
) -> str:
    """One hourly row: the variables given measured, the rest ``NaN``, then what places the row."""
    measured = measured or {}
    cells = [measured.get(channel.name, "NaN") for channel in SERIES]
    # Whole hours render as the files write them; a fraction is what a malformed file would carry.
    return "|".join([*cells, *descriptors, f"{hour:g}", label])


def stay(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


def with_stays(root: Path, files: Mapping[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / f"{name}.psv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def read_one(tmp_path: Path, content: bytes) -> Physionet2019CorpusReader:
    root = with_stays(tmp_path, {"training_setA/p000001": content})
    reader = Physionet2019CorpusReader(root, subsets=("training_setA",))
    reader.describe()
    return reader


@pytest.fixture
def reader() -> Physionet2019CorpusReader:
    return Physionet2019CorpusReader(SAMPLE)


def test_channels_are_the_clinical_variables_and_the_descriptors(
    reader: Physionet2019CorpusReader,
) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == TIMED + 3 + 2
    assert {channel.name for channel in schema if not channel.timeless} == {
        channel.name for channel in SERIES
    }
    assert {channel.name for channel in schema if channel.timeless} == {
        "Age",
        "Gender",
        "HospAdmTime",
        "Unit/MICU",
        "Unit/SICU",
    }
    assert "SepsisLabel" not in schema.names
    assert "ICULOS" not in schema.names


def test_an_hourly_stream_with_most_cells_empty_is_an_irregular_regime(
    reader: Physionet2019CorpusReader,
) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_stays_are_units_spanning_their_recorded_hours(
    reader: Physionet2019CorpusReader,
) -> None:
    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == list(SAMPLE_STAYS)
    assert [unit.extent for unit in units] == list(SAMPLE_EXTENTS)
    assert [sum(1 for _ in reader.read_observations(unit.key)) for unit in units] == list(
        SAMPLE_OBSERVATIONS
    )
    assert reader.describe().content.unit_count == len(SAMPLE_STAYS)
    assert reader.describe().content.observation_count == sum(SAMPLE_OBSERVATIONS)


def test_descriptors_become_timeless_features_and_an_unknown_ward_none(
    reader: Physionet2019CorpusReader,
) -> None:
    by_key = {str(unit.key): unit for unit in reader.read_units()}

    assert by_key["training_setA/p000001"].static_features == (
        StaticFeature("Age", 83.14),
        StaticFeature("Gender", 0.0),
        StaticFeature("HospAdmTime", -0.03),
    )
    assert by_key["training_setA/p000022"].static_features == (
        StaticFeature("Age", 77.26),
        StaticFeature("Gender", 0.0),
        StaticFeature("HospAdmTime", -135.81),
        StaticFeature("Unit/SICU", 1.0),
    )


def test_the_label_is_read_as_a_check_and_never_as_a_value(
    reader: Physionet2019CorpusReader,
) -> None:
    # The septic stay of the sample: ten of its rows carry the label, none of them a channel.
    observations = list(reader.read_observations(UnitKey("training_setA/p000015")))

    assert {observation.channel for observation in observations} <= {
        channel.name for channel in SERIES
    }


def test_checksum_is_that_of_the_file_bytes(reader: Physionet2019CorpusReader) -> None:
    expected = Checksum.of_bytes(
        b"".join((SAMPLE / f"{key}.psv").read_bytes() for key in SAMPLE_STAYS)
    )

    assert reader.describe().content.checksum == expected


def test_the_sample_keeps_the_line_endings_the_publisher_wrote() -> None:
    for key in SAMPLE_STAYS:
        content = (SAMPLE / f"{key}.psv").read_bytes()
        assert b"\r" not in content
        assert content.endswith(b"\n")


def test_sets_are_read_in_canonical_order_however_they_are_named() -> None:
    reversed_reader = Physionet2019CorpusReader(SAMPLE, subsets=tuple(reversed(SUBSETS)))

    assert reversed_reader.describe() == Physionet2019CorpusReader(SAMPLE).describe()
    assert [str(unit.key) for unit in reversed_reader.read_units()] == list(SAMPLE_STAYS)


@pytest.mark.parametrize("subsets", [(), ("training_setC",), ("training_setA", "set-b")])
def test_rejects_unknown_or_no_sets(subsets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="PhysioNet set"):
        Physionet2019CorpusReader(SAMPLE, subsets=subsets)


def test_a_missing_set_is_reported_as_missing_data(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="training_setA is missing"):
        Physionet2019CorpusReader(tmp_path).describe()


def test_a_set_without_stays_is_reported_as_missing_data(tmp_path: Path) -> None:
    (tmp_path / "training_setA").mkdir()

    with pytest.raises(CorpusDataNotFoundError, match="holds no stay"):
        Physionet2019CorpusReader(tmp_path, subsets=("training_setA",)).describe()


def test_excluded_stays_leave_the_corpus_whole_in_counts_checksum_and_keys() -> None:
    whole = Physionet2019CorpusReader(SAMPLE)
    cut = Physionet2019CorpusReader(SAMPLE, excluded=("p000015",))

    described = cut.describe()

    assert described.content.unit_count == whole.describe().content.unit_count - 1
    assert described.content.observation_count == (
        whole.describe().content.observation_count - SAMPLE_OBSERVATIONS[1]
    )
    assert described.content.checksum != whole.describe().content.checksum
    assert described.content.checksum == Checksum.of_bytes(
        b"".join((SAMPLE / f"{key}.psv").read_bytes() for key in SAMPLE_STAYS if "15" not in key)
    )
    assert [str(unit.key) for unit in cut.read_units()] == [
        key for key in SAMPLE_STAYS if not key.endswith("p000015")
    ]
    with pytest.raises(UnknownUnitError, match="not a stay of a selected set"):
        cut.read_observations(UnitKey("training_setA/p000015"))


def test_an_exclusion_that_matches_no_stay_is_refused_rather_than_ignored() -> None:
    cut = Physionet2019CorpusReader(SAMPLE, excluded=("p000015", "p999999"))

    with pytest.raises(CorpusDataNotFoundError, match=r"no selected set holds: \['p999999'\]"):
        cut.describe()
    with pytest.raises(CorpusDataNotFoundError, match="p999999"):
        list(cut.read_units())


def test_an_exclusion_of_a_stay_in_a_set_not_selected_is_refused() -> None:
    cut = Physionet2019CorpusReader(SAMPLE, subsets=("training_setA",), excluded=("p100006",))

    with pytest.raises(CorpusDataNotFoundError, match="p100006"):
        cut.describe()


@pytest.mark.parametrize("name", ["", " p000015", "p000015 "])
def test_an_exclusion_without_a_clean_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        Physionet2019CorpusReader(SAMPLE, excluded=(name,))


def test_a_stay_whose_hours_start_late_spans_from_its_first_row(tmp_path: Path) -> None:
    reader = read_one(tmp_path, stay(row(5), row(6, {"HR": "80"})))

    (unit,) = reader.read_units()

    assert unit.extent == TimeExtent(5.0, 7.0)
    assert list(reader.read_observations(unit.key)) == [Observation("HR", 6.0, 80.0)]


def test_a_stay_of_empty_cells_is_a_unit_that_measures_nothing(tmp_path: Path) -> None:
    root = with_stays(
        tmp_path,
        {
            "training_setA/p000001": stay(row(1), row(2)),
            "training_setA/p000002": stay(row(1, {"HR": "80"})),
        },
    )
    reader = Physionet2019CorpusReader(root, subsets=("training_setA",))

    empty, _ = reader.read_units()

    assert empty.extent == TimeExtent(1.0, 3.0)
    assert list(reader.read_observations(empty.key)) == []
    assert reader.describe().content == reader.describe().content
    assert (reader.describe().content.unit_count, reader.describe().content.observation_count) == (
        2,
        1,
    )


def test_a_descriptor_recorded_as_unknown_is_no_feature(tmp_path: Path) -> None:
    reader = read_one(
        tmp_path, stay(row(1, {"HR": "80"}, descriptors=("NaN", "NaN", "NaN", "NaN", "NaN")))
    )

    (unit,) = reader.read_units()

    assert unit.static_features == ()


def test_a_ward_flagged_present_is_a_token_and_one_flagged_absent_is_not(tmp_path: Path) -> None:
    reader = read_one(tmp_path, stay(row(1, {"HR": "80"}, descriptors=("54", "1", "1", "0", "-2"))))

    (unit,) = reader.read_units()

    assert unit.static_features == (
        StaticFeature("Age", 54.0),
        StaticFeature("Gender", 1.0),
        StaticFeature("HospAdmTime", -2.0),
        StaticFeature("Unit/MICU", 1.0),
    )


def test_every_measured_cell_of_an_hour_is_an_observation_at_that_hour(tmp_path: Path) -> None:
    reader = read_one(tmp_path, stay(row(3, {"HR": "80", "Platelets": "150"}), row(4)))

    (unit,) = reader.read_units()

    assert list(reader.read_observations(unit.key)) == [
        Observation("HR", 3.0, 80.0),
        Observation("Platelets", 3.0, 150.0),
    ]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "no header"),
        (stay(header=HEADER.replace("HR|", "HeartRate|")), "expected the header"),
        (stay(), "no row"),
        (stay(row(1) + "|extra"), "expected 41 columns"),
        (stay(row(1, {"HR": "eighty"})), "could not convert"),
        (stay(row(1, {"HR": "inf"})), "non-finite"),
        (stay(row(2), row(2)), "goes back in time"),
        (stay(row(2), row(1)), "goes back in time"),
        (stay(row(0.5)), "not a whole number of hours"),
        (stay(row(-1)), "not a whole number of hours"),
        (stay(row(1, label="2")), "SepsisLabel is not a bit"),
        (stay(row(1), row(2, descriptors=("55", "0", "NaN", "NaN", "-0.03"))), "changes within"),
        (stay(row(1, descriptors=("54", "0", "1", "1", "-0.03"))), "both wards"),
        (stay(row(1, descriptors=("54", "0", "2", "0", "-0.03"))), "not a bit"),
        (stay(row(1, descriptors=("old", "0", "NaN", "NaN", "-0.03"))), "could not convert"),
    ],
)
def test_rejects_malformed_files(tmp_path: Path, content: bytes, reason: str) -> None:
    with pytest.raises(MalformedCorpusDataError, match=reason) as caught:
        read_one(tmp_path, content)

    assert "training_setA/p000001.psv" in str(caught.value)


def test_blank_lines_and_carriage_returns_do_not_change_what_is_read(tmp_path: Path) -> None:
    content = stay(row(1, {"HR": "80"}), row(2))
    reader = read_one(tmp_path, content.replace(b"\n", b"\r\n\n"))

    assert reader.describe().content.observation_count == 1


@pytest.mark.parametrize(
    "key",
    [
        "p000001",
        "training_setA/p999999",
        "training_setC/p000001",
        "training_setA/../training_setA/p000001",
        "training_setA/training_setB/p100006",
    ],
)
def test_a_key_that_could_name_no_stay_is_unknown_before_the_stream(
    reader: Physionet2019CorpusReader, key: str
) -> None:
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey(key))


def test_reading_a_stay_from_a_missing_set_is_missing_data(tmp_path: Path) -> None:
    with_stays(tmp_path, {"training_setA/p000001": stay(row(1))})

    with pytest.raises(CorpusDataNotFoundError):
        Physionet2019CorpusReader(tmp_path).read_observations(UnitKey("training_setB/p100001"))


@pytest.mark.skipif(
    raw_root("physionet2019") is None, reason="the raw corpus is not on this machine"
)
def test_both_published_sets_describe_as_many_observations_as_they_stream() -> None:
    reader = Physionet2019CorpusReader(raw_root("physionet2019") or Path())

    described = reader.describe().content
    units = list(reader.read_units())

    assert described.unit_count == len(units) == 20_336 + 20_000
    assert described.observation_count == sum(
        1 for unit in units for _ in reader.read_observations(unit.key)
    )
