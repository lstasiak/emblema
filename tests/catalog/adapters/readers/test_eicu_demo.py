"""The eICU demo reader reads a unit stay as 2012 frames one, in the 2012 variables.

Each rule is held to tables built by hand in the database's layout, whose readings are known; the
miniature sample of real stays is read whole; the downloaded demo is read where it was fetched.
"""

import csv
import gzip
import io
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.eicu_demo import EicuDemoCorpusReader
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.corpora import raw_root, sample

PATIENT = (
    "patientunitstayid",
    "hospitalid",
    "gender",
    "age",
    "admissionheight",
    "admissionweight",
    "unittype",
    "unitdischargeoffset",
)
NURSE = (
    "patientunitstayid",
    "nursingchartoffset",
    "nursingchartcelltypecat",
    "nursingchartcelltypevalname",
    "nursingchartvalue",
)
LAB = ("patientunitstayid", "labresultoffset", "labname", "labresult", "labresultrevisedoffset")
OUTPUT = ("patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric")
URINE = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine"
STAY = UnitKey("7/1")


def patient(
    stay: str = "1",
    *,
    hospital: str = "7",
    gender: str = "Male",
    age: str = "60",
    height: str = "180",
    weight: str = "",
    unit_type: str = "MICU",
    discharge: str = "4000",
) -> tuple[str, ...]:
    return stay, hospital, gender, age, height, weight, unit_type, discharge


def charted(
    offset: int, name: str = "Heart Rate", value: str = "80", *, stay: str = "1"
) -> tuple[str, ...]:
    return stay, str(offset), "Vital Signs", name, value


def lab(
    offset: int, name: str = "sodium", value: str = "140", revised: str = "", *, stay: str = "1"
) -> tuple[str, ...]:
    return stay, str(offset), name, value, revised


def output(
    offset: int, label: str = "Urine", value: str = "50", *, path: str = URINE, stay: str = "1"
) -> tuple[str, ...]:
    return stay, str(offset), path, label, value


def table(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows([header, *rows])
    path.write_bytes(gzip.compress(text.getvalue().encode("utf-8"), mtime=0))


def demo(
    root: Path,
    *,
    patients: list[tuple[str, ...]] | None = None,
    nurse: list[tuple[str, ...]] | None = None,
    labs: list[tuple[str, ...]] | None = None,
    outputs: list[tuple[str, ...]] | None = None,
) -> EicuDemoCorpusReader:
    """A reader of the four tables, one stay of hospital 7 unless patients are given."""
    root.mkdir(parents=True, exist_ok=True)
    table(root / "patient.csv.gz", PATIENT, patients if patients is not None else [patient()])
    table(root / "nurseCharting.csv.gz", NURSE, nurse or [])
    table(root / "lab.csv.gz", LAB, labs or [])
    table(root / "intakeOutput.csv.gz", OUTPUT, outputs or [])
    return EicuDemoCorpusReader(root)


def observations(reader: EicuDemoCorpusReader, key: UnitKey = STAY) -> list[Observation]:
    return list(reader.read_observations(key))


def test_a_charted_variable_is_read_as_its_2012_name_in_hours(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        nurse=[
            charted(90, "Non-Invasive BP Systolic", "120"),
            charted(30, "Temperature (C)", "37.2"),
            charted(30, "Temperature (F)", "99"),
        ],
    )

    assert observations(reader) == [
        Observation("Temp", 0.5, 37.2),
        Observation("NISysABP", 1.5, 120.0),
    ]


def test_a_charted_cell_outside_the_vital_signs_and_scores_is_not_read(tmp_path: Path) -> None:
    other = ("1", "10", "Other Vital Signs and Infusions", "Heart Rate", "80")
    scored = ("1", "20", "Scores", "GCS Total", "15")

    reader = demo(tmp_path, nurse=[other, scored])

    assert observations(reader) == [Observation("GCS", 20 / 60, 15.0)]


def test_a_cell_holding_no_finite_number_is_no_reading(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        nurse=[charted(10, value="Unable to score"), charted(20, value="nan")],
        labs=[lab(30, value="")],
        outputs=[output(40, value="")],
    )

    assert observations(reader) == []


def test_a_reading_is_the_stays_from_admission_to_hour_48(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        nurse=[charted(-5), charted(0), charted(2880), charted(2881)],
        labs=[lab(-300)],
    )

    assert [o.time for o in observations(reader)] == [0.0, 48.0]


def test_a_reading_after_the_patient_left_the_unit_is_not_the_stays(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        patients=[patient(discharge="600")],
        nurse=[charted(600), charted(601)],
        labs=[lab(900)],
    )

    assert observations(reader) == [Observation("HR", 10.0, 80.0)]
    assert next(reader.read_units()).extent.end == 48.0 + 1 / 60


def test_a_stay_that_leaves_before_it_arrives_is_refused(tmp_path: Path) -> None:
    reader = demo(tmp_path, patients=[patient(discharge="-5")])

    with pytest.raises(MalformedCorpusDataError, match="leaves before it arrives"):
        reader.describe()


def test_the_latest_revision_of_a_laboratory_result_is_kept(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        labs=[
            lab(60, value="138", revised="70"),
            lab(60, value="141", revised="200"),
            lab(60, value="139", revised="90"),
            lab(120, "potassium", "4.0"),
            lab(120, "potassium", "4.2", revised="-30"),
        ],
    )

    assert observations(reader) == [Observation("Na", 1.0, 141.0), Observation("K", 2.0, 4.2)]


def test_of_two_readings_alike_in_all_else_the_first_row_is_kept(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        labs=[lab(60, "potassium", "4.1"), lab(60, "potassium", "4.4")],
        outputs=[
            output(120, "Urine", "300"),
            output(120, "Urinary Catheter Output: Indwelling/Continuous Ure", "550"),
        ],
    )

    assert observations(reader) == [
        Observation("K", 1.0, 4.1),
        Observation("Urine", 2.0, 300.0),
    ]


@pytest.mark.parametrize(
    "label",
    [
        "Urine",
        "URINE CATHETER",
        "Foley cath",
        "Indwelling Catheter Output",
        "Output (mL)-Urethral Catheter Non-latex 16 Fr.",
        "Voided Amount",
        "Nephrostomy",
    ],
)
def test_urine_output_is_a_volume_under_the_output_cells(tmp_path: Path, label: str) -> None:
    reader = demo(tmp_path, outputs=[output(60, label, "100")])

    assert observations(reader) == [Observation("Urine", 1.0, 100.0)]


@pytest.mark.parametrize(
    ("label", "path"),
    [
        ("Urine Count", URINE),
        ("Urine Occurrence", URINE),
        ("Mixed Urine/Stool Volume", URINE),
        ("Unmeasured voids", URINE),
        ("Stool", URINE),
        ("Urine", URINE.replace("Output", "Intake")),
    ],
)
def test_a_cell_that_is_no_volume_of_urine_is_not_read(
    tmp_path: Path, label: str, path: str
) -> None:
    reader = demo(tmp_path, outputs=[output(60, label, "100", path=path)])

    assert observations(reader) == []


def test_a_laboratory_result_the_2012_stays_do_not_have_is_not_read(tmp_path: Path) -> None:
    reader = demo(
        tmp_path, labs=[lab(60, "bedside glucose", "110"), lab(60, "HCO3", "24"), lab(60, "RBC")]
    )

    assert observations(reader) == []


def test_the_fraction_of_inspired_oxygen_is_read_in_percent(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        labs=[
            lab(60, "FiO2", "40"),
            lab(120, "FiO2", "0.5"),
            lab(180, "FiO2", "1"),
            lab(240, value="1"),
        ],
    )

    assert [o.value for o in observations(reader)] == [40.0, 50.0, 100.0, 1.0]


def test_the_weight_at_admission_is_the_first_reading_of_the_weight_series(
    tmp_path: Path,
) -> None:
    reader = demo(
        tmp_path,
        patients=[patient(weight="81.5"), patient("2", weight="")],
        nurse=[charted(0)],
    )

    assert observations(reader) == [Observation("HR", 0.0, 80.0), Observation("Weight", 0.0, 81.5)]
    assert observations(reader, UnitKey("7/2")) == []


def test_the_descriptors_are_timeless_tokens_as_the_2012_stays_carry_them(tmp_path: Path) -> None:
    reader = demo(
        tmp_path,
        patients=[
            patient("1", gender="Female", age="> 89", height="162.5", unit_type="Med-Surg ICU"),
            patient("2", gender="Unknown", age="", height="", unit_type="CSICU"),
        ],
    )

    first, second = reader.read_units()

    assert first.static_features == (
        StaticFeature("Age", 90.0),
        StaticFeature("Gender", 0.0),
        StaticFeature("Height", 162.5),
        StaticFeature("UnitType/Med-Surg ICU", 1.0),
    )
    assert second.static_features == (StaticFeature("UnitType/CSICU", 1.0),)


def test_a_stay_is_keyed_within_its_hospital_and_spans_the_first_48_hours(tmp_path: Path) -> None:
    reader = demo(tmp_path, patients=[patient("12", hospital="3"), patient("4", hospital="20")])

    units = list(reader.read_units())

    assert [unit.key for unit in units] == [UnitKey("20/4"), UnitKey("3/12")]
    assert {(unit.extent.start, unit.extent.end) for unit in units} == {(0.0, 48.0 + 1 / 60)}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unit_type": "Burn Unit"}, "no unit type"),
        ({"age": "sixty"}, "not an age"),
        ({"hospital": ""}, "non-blank"),
    ],
)
def test_a_descriptor_the_database_does_not_write_is_refused(
    tmp_path: Path, change: dict[str, str], message: str
) -> None:
    reader = demo(tmp_path, patients=[patient(**change)])

    with pytest.raises(MalformedCorpusDataError, match=message):
        reader.describe()


def test_a_stay_given_twice_is_refused(tmp_path: Path) -> None:
    reader = demo(tmp_path, patients=[patient(), patient()])

    with pytest.raises(MalformedCorpusDataError, match="given twice"):
        reader.describe()


def test_a_reading_of_a_stay_no_patient_has_is_refused(tmp_path: Path) -> None:
    reader = demo(tmp_path, nurse=[charted(10, stay="2")])

    with pytest.raises(MalformedCorpusDataError, match="no patient"):
        reader.describe()


@pytest.mark.parametrize("offset", ["ten", " 12", "+12", "1_000", "١٢", "1.5"])
@pytest.mark.parametrize("where", ["nurse", "revision", "discharge"])
def test_an_offset_that_is_no_whole_minute_in_ascii_digits_is_refused(
    tmp_path: Path, offset: str, where: str
) -> None:
    by_table: dict[str, dict[str, list[tuple[str, ...]]]] = {
        "nurse": {"nurse": [("1", offset, "Vital Signs", "Heart Rate", "80")]},
        "revision": {"labs": [lab(10, revised=offset)]},
        "discharge": {"patients": [patient(discharge=offset)]},
    }
    reader = demo(tmp_path, **by_table[where])

    with pytest.raises(MalformedCorpusDataError, match="is not an offset in minutes"):
        reader.describe()


def test_a_missing_table_is_reported_by_name(tmp_path: Path) -> None:
    demo(tmp_path)
    (tmp_path / "lab.csv.gz").unlink()

    with pytest.raises(CorpusDataNotFoundError, match=r"lab\.csv\.gz"):
        EicuDemoCorpusReader(tmp_path).describe()


def test_a_table_without_a_column_read_is_refused(tmp_path: Path) -> None:
    demo(tmp_path)
    table(tmp_path / "lab.csv.gz", LAB[:-1], [])

    with pytest.raises(MalformedCorpusDataError, match="labresultrevisedoffset"):
        EicuDemoCorpusReader(tmp_path).describe()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"patientunitstayid\n", "not a gzipped file"),
        (
            gzip.compress(",".join(PATIENT).encode() + b"\n1,7,Male,60,180,,MICU,4000\n")[:-8],
            "gzipped",
        ),
        (gzip.compress(b"\xff\xfe\x00broken"), "not a UTF-8 CSV table"),
        (gzip.compress(",".join(PATIENT).encode() + b'\n"' + b"x" * 200_000 + b'"\n'), "CSV"),
    ],
    ids=["plain", "truncated", "not-utf8", "field-too-large"],
)
def test_a_table_that_is_not_a_gzipped_csv_is_refused(
    tmp_path: Path, content: bytes, message: str
) -> None:
    demo(tmp_path)
    (tmp_path / "patient.csv.gz").write_bytes(content)

    with pytest.raises(MalformedCorpusDataError, match=message):
        EicuDemoCorpusReader(tmp_path).describe()


def test_a_stay_without_a_reading_is_still_a_unit(tmp_path: Path) -> None:
    reader = demo(tmp_path, patients=[patient(), patient("2")], nurse=[charted(10, stay="2")])

    assert reader.describe().content.unit_count == 2
    assert observations(reader) == []
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey("7/3"))


def test_describing_again_reads_the_tables_as_they_now_are(tmp_path: Path) -> None:
    reader = demo(tmp_path, nurse=[charted(10)])
    before = reader.describe()
    table(tmp_path / "nurseCharting.csv.gz", NURSE, [charted(10), charted(20)])

    after = reader.describe()

    assert after.content.checksum != before.content.checksum
    assert after.content.observation_count == 2


def test_the_sample_is_read_whole_in_the_2012_variables() -> None:
    reader = EicuDemoCorpusReader(sample("eicu_demo"))

    description = reader.describe()
    units = list(reader.read_units())

    assert description.sampling_regime is SamplingRegime.IRREGULAR
    assert description.channel_schema == EicuDemoCorpusReader.SCHEMA
    assert (description.content.unit_count, description.content.observation_count) == (5, 179)
    assert [str(unit.key) for unit in units] == [
        "120/376041",
        "268/1817131",
        "420/3035602",
        "93/258915",
        "93/301185",
    ]
    assert observations(reader, UnitKey("420/3035602")) == []


@pytest.mark.skipif(raw_root("eicu_demo") is None, reason="the raw corpus is not on this machine")
def test_the_downloaded_demo_is_read_whole() -> None:
    root = raw_root("eicu_demo")
    assert root is not None
    reader = EicuDemoCorpusReader(root)

    content = reader.describe().content

    assert (content.unit_count, content.observation_count) == (2520, 510_050)
