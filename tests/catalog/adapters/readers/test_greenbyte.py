import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.greenbyte import (
    ORIGIN_TEXT,
    GreenbyteCorpusReader,
    WindFarm,
)
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
from tests.support.corpora import raw_root, sample

pytest.importorskip("pyarrow")

KELMARSH = GreenbyteCorpusReader.KELMARSH
PENMANSHIEL = GreenbyteCorpusReader.PENMANSHIEL
SIGNALS = GreenbyteCorpusReader.SIGNALS

SAMPLE = sample("kelmarsh")
# A farm of two turbines, so a test writes a table or two rather than six.
FARM = WindFarm("Kelmarsh", ("1", "2"))
POWER = "Power (kW)"
OIL = "Gear oil temperature (°C)"
PREAMBLE = (
    "# This file was exported by Greenbyte at 2022-01-27 10:33:05.",
    "#",
    "# Turbine: Kelmarsh 1",
    "# Turbine type: Senvion MM92",
    "# Time zone: UTC",
    "# Time interval: 2016-01-01 00:00:00 - 2017-01-01 00:00:00 (366 days)",
    "#",
    '# Data that is missing or is erroneous has been marked with the value "NaN"',
    "#",
)
# A row is its stamp and one cell per column of the table, as text.
Row = tuple[str, ...]


def table(
    rows: Sequence[Row],
    columns: Sequence[str] = (POWER, OIL),
    preamble: Sequence[str] = PREAMBLE,
) -> bytes:
    """A turbine table as the export writes it: preamble, header, a row per instant."""
    header = ",".join(["# Date and time", *(f'"{column}"' for column in columns)])
    lines = [*preamble, header, *(",".join(row) for row in rows)]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def member(turbine: str, year: int = 2016, farm: str = "Kelmarsh") -> str:
    return f"Turbine_Data_{farm}_{turbine}_{year}-01-01_-_{year + 1}-01-01_228.csv"


def with_archive(root: Path, name: str, members: Mapping[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in members.items():
            archive.writestr(path, content)
    (root / name).write_bytes(buffer.getvalue())
    return root


def farm_of(root: Path, tables: Mapping[str, bytes], name: str = "Kelmarsh_SCADA_2016.zip") -> Path:
    return with_archive(root, name, tables)


def reader_of(root: Path, *turbines: str, farm: WindFarm = FARM) -> GreenbyteCorpusReader:
    return GreenbyteCorpusReader(root, farm, turbines)


def readings(reader: GreenbyteCorpusReader) -> list[Observation]:
    return [
        observation
        for unit in reader.read_units()
        for observation in reader.read_observations(unit.key)
    ]


def hours(stamp: str) -> float:
    """A stamp of 2016 as hours since the origin: ``MM-DD HH:MM``."""
    month, day, hour, minute = map(int, stamp.replace("-", " ").replace(":", " ").split())
    days = sum((31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[: month - 1]) + day - 1
    return days * 24.0 + hour + minute / 60.0


@pytest.fixture
def reader() -> GreenbyteCorpusReader:
    return GreenbyteCorpusReader(SAMPLE, KELMARSH, ("1", "2"))


def test_channels_are_the_sixty_signals_of_the_manufacturers_mapping(
    reader: GreenbyteCorpusReader,
) -> None:
    schema = reader.describe().channel_schema

    assert len(schema) == len(SIGNALS) == 60
    assert Channel("Gear oil temperature", "°C") in schema.channels
    assert Channel("Power", "kW") in schema.channels
    assert Channel("Power factor (cosphi)") in schema.channels


def test_a_fixed_cadence_is_a_regular_regime(reader: GreenbyteCorpusReader) -> None:
    assert reader.describe().sampling_regime is SamplingRegime.REGULAR


def test_the_checksum_opens_with_the_selection_and_covers_every_archive_in_name_order(
    tmp_path: Path,
) -> None:
    rows = [("2016-03-01 00:00:00", "1.5", "40.0")]
    root = with_archive(tmp_path, "Kelmarsh_SCADA_2017.zip", {member("1", 2017): table(rows)})
    farm_of(root, {member("1"): table(rows)})

    checksum = reader_of(root, "1").describe().content.checksum

    statement = (
        f"greenbyte: farm Kelmarsh; turbines 1; averages of "
        f"{'|'.join(signal.name for signal in SIGNALS)}; origin {ORIGIN_TEXT}\n"
    ).encode()
    first = (root / "Kelmarsh_SCADA_2016.zip").read_bytes()
    second = (root / "Kelmarsh_SCADA_2017.zip").read_bytes()
    assert checksum == Checksum.of_bytes(statement + first + second)


def test_another_selection_of_turbines_is_another_checksum(tmp_path: Path) -> None:
    rows = [("2016-03-01 00:00:00", "1.5", "40.0")]
    root = farm_of(tmp_path, {member("1"): table(rows), member("2"): table(rows)})

    assert (
        reader_of(root, "1").describe().content.checksum
        != reader_of(root, "1", "2").describe().content.checksum
    )


def test_all_turbines_are_read_unless_some_are_named(tmp_path: Path) -> None:
    rows = [("2016-03-01 00:00:00", "1.5", "40.0")]
    root = farm_of(tmp_path, {member("2"): table(rows), member("1"): table(rows)})

    assert [str(unit.key) for unit in reader_of(root).read_units()] == ["1/2016-03", "2/2016-03"]
    assert [str(unit.key) for unit in reader_of(root, "2").read_units()] == ["2/2016-03"]


def test_only_the_averages_of_the_mapped_signals_are_read(tmp_path: Path) -> None:
    columns = (
        POWER,
        "Power, Standard deviation (kW)",
        "Lost Production Total (kWh)",
        OIL,
        "Gear oil temperature, Max (°C)",
    )
    rows = [("2016-03-01 00:00:00", "1.5", "0.2", "9.0", "40.0", "44.0")]
    root = farm_of(tmp_path, {member("1"): table(rows, columns)})

    assert readings(reader_of(root, "1")) == [
        Observation("Gear oil temperature", hours("03-01 00:00"), 40.0),
        Observation("Power", hours("03-01 00:00"), 1.5),
    ]


def test_an_empty_cell_or_a_nan_is_no_reading(tmp_path: Path) -> None:
    rows = [
        ("2016-03-01 00:00:00", "NaN", "40.0"),
        ("2016-03-01 00:10:00", "1.5", ""),
        ("2016-03-01 00:20:00", "NaN", "NaN"),
    ]
    root = farm_of(tmp_path, {member("1"): table(rows)})
    reader = reader_of(root, "1")

    assert reader.describe().content.observation_count == 2
    assert [(o.channel, o.value) for o in readings(reader)] == [
        ("Gear oil temperature", 40.0),
        ("Power", 1.5),
    ]


def test_an_average_lies_at_its_stamp_in_hours_since_the_origin(tmp_path: Path) -> None:
    rows = [("2016-01-01 00:00:00", "1.0", "NaN"), ("2016-01-02 06:30:00", "2.0", "NaN")]
    root = farm_of(tmp_path, {member("1"): table(rows)})

    assert [o.time for o in readings(reader_of(root, "1"))] == [0.0, 30.5]


def test_months_with_a_reading_are_units_cut_to_the_whole_hours_around_the_span(
    tmp_path: Path,
) -> None:
    rows = [
        ("2016-01-23 12:10:00", "1.0", "NaN"),
        ("2016-01-23 12:20:00", "1.0", "NaN"),
        ("2016-03-05 07:50:00", "1.0", "NaN"),
    ]
    root = farm_of(tmp_path, {member("1"): table(rows)})

    units = list(reader_of(root, "1").read_units())

    # February holds no reading and is no unit; the span opens at noon and closes at eight.
    assert [(str(unit.key), unit.extent) for unit in units] == [
        ("1/2016-01", TimeExtent(hours("01-23 12:00"), hours("02-01 00:00"))),
        ("1/2016-03", TimeExtent(hours("03-01 00:00"), hours("03-05 08:00"))),
    ]


def test_a_month_between_the_first_and_the_last_spans_the_whole_calendar_month(
    tmp_path: Path,
) -> None:
    rows = [
        ("2016-01-23 12:10:00", "1.0", "NaN"),
        ("2016-02-10 00:00:00", "1.0", "NaN"),
        ("2016-03-05 07:50:00", "1.0", "NaN"),
    ]
    root = farm_of(tmp_path, {member("1"): table(rows)})

    february = list(reader_of(root, "1").read_units())[1]

    assert february.extent == TimeExtent(hours("02-01 00:00"), hours("03-01 00:00"))


def test_a_cumulative_export_yields_each_instant_once(tmp_path: Path) -> None:
    # Every block restarts the period and repeats the instants before it, holding values in its
    # newest rows alone.
    rows = [
        ("2016-03-01 00:00:00", "1.0", "40.0"),
        ("2016-03-01 00:00:00", "NaN", "NaN"),
        ("2016-03-01 00:10:00", "2.0", "41.0"),
        ("2016-03-01 00:00:00", "NaN", "NaN"),
        ("2016-03-01 00:10:00", "NaN", "NaN"),
        ("2016-03-01 00:20:00", "3.0", "42.0"),
    ]
    root = farm_of(tmp_path, {member("1"): table(rows)})
    reader = reader_of(root, "1")

    assert reader.describe().content.observation_count == 6
    assert [(o.time, o.channel, o.value) for o in readings(reader)] == [
        (hours("03-01 00:00"), "Gear oil temperature", 40.0),
        (hours("03-01 00:00"), "Power", 1.0),
        (hours("03-01 00:10"), "Gear oil temperature", 41.0),
        (hours("03-01 00:10"), "Power", 2.0),
        (hours("03-01 00:20"), "Gear oil temperature", 42.0),
        (hours("03-01 00:20"), "Power", 3.0),
    ]


def test_rows_of_one_instant_may_each_hold_another_signal(tmp_path: Path) -> None:
    rows = [("2016-03-01 00:00:00", "1.0", "NaN"), ("2016-03-01 00:00:00", "NaN", "40.0")]
    root = farm_of(tmp_path, {member("1"): table(rows)})

    assert [(o.channel, o.value) for o in readings(reader_of(root, "1"))] == [
        ("Gear oil temperature", 40.0),
        ("Power", 1.0),
    ]


def test_a_value_repeated_alike_is_one_reading_and_one_repeated_otherwise_is_refused(
    tmp_path: Path,
) -> None:
    alike = [("2016-03-01 00:00:00", "1.0", "NaN"), ("2016-03-01 00:00:00", "1.0", "NaN")]
    otherwise = [("2016-03-01 00:00:00", "1.0", "NaN"), ("2016-03-01 00:00:00", "1.5", "NaN")]

    agreed = reader_of(farm_of(tmp_path / "a", {member("1"): table(alike)}), "1")
    assert agreed.describe().content.observation_count == 1
    with pytest.raises(MalformedCorpusDataError, match="two rows of 2016-03-01T00:00:00 disagree"):
        reader_of(farm_of(tmp_path / "b", {member("1"): table(otherwise)}), "1").describe()


def test_a_unit_spelt_anew_from_one_table_to_the_next_stays_one_channel(tmp_path: Path) -> None:
    # As the exports hold it: the old spelling through 2022, the new one from 2023.
    old = table([("2016-12-31 23:50:00", "7.0")], ("Drive train acceleration (mm/ss)",))
    new = table([("2017-01-01 00:00:00", "8.0")], ("Drive train acceleration (mm/s2)",))
    root = farm_of(tmp_path, {member("1"): old})
    with_archive(root, "Kelmarsh_SCADA_2017.zip", {member("1", 2017): new})

    reader = reader_of(root, "1")

    assert {o.channel for o in readings(reader)} == {"Drive train acceleration"}
    assert [o.value for o in readings(reader)] == [7.0, 8.0]
    assert Channel("Drive train acceleration", "mm/s2") in reader.describe().channel_schema.channels


def test_an_instant_repeated_across_tables_follows_the_same_rule(tmp_path: Path) -> None:
    first = table([("2016-12-31 23:50:00", "1.0", "NaN")])
    alike = table([("2016-12-31 23:50:00", "1.0", "40.0"), ("2017-01-01 00:00:00", "2.0", "NaN")])
    otherwise = table([("2016-12-31 23:50:00", "1.5", "NaN")])
    agreed = farm_of(tmp_path / "a", {member("1"): first})
    with_archive(agreed, "Kelmarsh_SCADA_2017.zip", {member("1", 2017): alike})
    disputed = farm_of(tmp_path / "b", {member("1"): first})
    with_archive(disputed, "Kelmarsh_SCADA_2017.zip", {member("1", 2017): otherwise})

    assert [(o.channel, o.value) for o in readings(reader_of(agreed, "1"))] == [
        ("Gear oil temperature", 40.0),
        ("Power", 1.0),
        ("Power", 2.0),
    ]
    with pytest.raises(
        MalformedCorpusDataError, match="two rows of 2016-12-31T23:50:00 disagree in Power"
    ):
        reader_of(disputed, "1").describe()


def test_a_table_longer_than_one_block_of_the_stream_is_read_whole(tmp_path: Path) -> None:
    # Sixty thousand instants, each on two rows as a cumulative export writes them: three
    # megabytes, so the stream hands the table over in several blocks and an instant's rows
    # fall in different ones.
    stamps = [
        f"2016-{1 + day // 28:02d}-{1 + day % 28:02d} {minute // 60:02d}:{minute % 60:02d}:00"
        for day in range(336)
        for minute in range(0, 1440, 8)
    ][:60_000]
    held = [(stamp, f"{index}.5", "NaN") for index, stamp in enumerate(stamps)]
    empty = [(stamp, "NaN", "NaN") for stamp in stamps]
    content = table([*held, *empty])
    assert len(content) > 3_000_000
    root = farm_of(tmp_path, {member("1"): content})
    reader = reader_of(root, "1")

    assert reader.describe().content.observation_count == 60_000
    values = [o.value for o in readings(reader)]
    assert values == [index + 0.5 for index in range(60_000)]


@pytest.mark.parametrize(
    ("signal", "column", "held"),
    [
        ("Power", "Power (kW)", True),
        ("Tower Acceleration Y", "Tower Acceleration y (mm/ss)", True),
        ("Power", "Power, Maximum (kW)", False),
        ("Power", "Power factor (cosphi)", False),
        ("Power", "Reactive power (kvar)", False),
        ("Power factor (cosphi)", "Power factor (cosphi)", True),
        ("Power factor (cosphi)", "Power factor (cosphi), Max", False),
        ("Ambient temperature (converter)", "Ambient temperature (converter) (°C)", True),
        ("Drive train acceleration", "Drive train acceleration (mm/ss)", True),
        ("Drive train acceleration", "Drive train acceleration (mm/s2)", True),
        ("Blade angle (pitch position) A", "Blade angle (pitch position) (°)", False),
        ("Wind speed", "Wind speed Sensor 1 (m/s)", False),
    ],
)
def test_a_column_is_a_signals_by_its_title_and_unit(signal: str, column: str, held: bool) -> None:
    named = next(each for each in SIGNALS if each.name == signal)

    assert named.holds(column) is held


def test_no_two_signals_share_a_title_whatever_its_case() -> None:
    titles = [signal.name.casefold() for signal in SIGNALS]

    assert len(set(titles)) == len(titles)


def test_a_turbines_rows_are_laid_in_time_order_however_its_tables_arrive(tmp_path: Path) -> None:
    # The archive read first holds the later year, and its table's rows run backwards.
    later = table([("2017-02-01 00:10:00", "3.0", "NaN"), ("2017-02-01 00:00:00", "2.0", "NaN")])
    earlier = table([("2016-03-01 00:00:00", "1.0", "NaN")])
    root = with_archive(tmp_path, "Kelmarsh_SCADA_a.zip", {member("1", 2017): later})
    with_archive(root, "Kelmarsh_SCADA_b.zip", {member("1"): earlier})

    reader = reader_of(root, "1")

    assert [str(unit.key) for unit in reader.read_units()] == ["1/2016-03", "1/2017-02"]
    assert [o.value for o in readings(reader)] == [1.0, 2.0, 3.0]


def test_status_logs_and_other_members_are_not_read(tmp_path: Path) -> None:
    root = farm_of(
        tmp_path,
        {
            member("1"): table([("2016-03-01 00:00:00", "1.0", "NaN")]),
            "Status_Kelmarsh_1_2016-01-01_-_2017-01-01_228.csv": b"Timestamp start,Status\r\n",
            "readme.txt": b"not a table",
        },
    )

    assert reader_of(root, "1").describe().content.observation_count == 1


def test_files_beside_the_farms_archives_are_not_read(tmp_path: Path) -> None:
    root = farm_of(tmp_path, {member("1"): table([("2016-03-01 00:00:00", "1.0", "NaN")])})
    (root / "Kelmarsh_PMU_4459.zip").write_bytes(b"not an archive of turbines")
    (root / ".unpacked-Kelmarsh_SCADA_2016.zip").write_bytes(b"")

    assert reader_of(root, "1").describe().content.unit_count == 1


@pytest.mark.parametrize("turbines", [("7",), ("1", "07")])
def test_a_turbine_the_farm_does_not_have_is_refused(turbines: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="unknown turbines"):
        GreenbyteCorpusReader(SAMPLE, KELMARSH, turbines)


def test_a_missing_directory_or_a_directory_without_archives_is_missing_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="is missing"):
        reader_of(tmp_path / "absent").describe()
    with pytest.raises(CorpusDataNotFoundError, match="no Kelmarsh_SCADA"):
        reader_of(tmp_path).describe()


def test_a_selected_turbine_without_a_table_is_missing_data(tmp_path: Path) -> None:
    root = farm_of(tmp_path, {member("1"): table([("2016-03-01 00:00:00", "1.0", "NaN")])})

    with pytest.raises(CorpusDataNotFoundError, match=r"no table of Kelmarsh turbines \['2'\]"):
        reader_of(root).describe()


def test_a_table_of_another_farm_or_of_no_turbine_of_this_one_is_malformed(
    tmp_path: Path,
) -> None:
    rows = [("2016-03-01 00:00:00", "1.0", "NaN")]
    other = farm_of(tmp_path / "a", {member("1", farm="Penmanshiel"): table(rows)})
    stray = farm_of(tmp_path / "b", {member("9"): table(rows)})

    with pytest.raises(MalformedCorpusDataError, match="a table of Penmanshiel"):
        reader_of(other, "1").describe()
    with pytest.raises(MalformedCorpusDataError, match="names turbine 9"):
        reader_of(stray, "1").describe()


def test_an_archive_that_is_no_archive_is_malformed(tmp_path: Path) -> None:
    (tmp_path / "Kelmarsh_SCADA_2016.zip").write_bytes(b"not a zip")

    with pytest.raises(MalformedCorpusDataError, match="not a readable zip archive"):
        reader_of(tmp_path, "1").describe()


ROWS = [("2016-03-01 00:00:00", "1.0", "40.0")]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (table(ROWS, preamble=[line for line in PREAMBLE if "Time zone" not in line]), "UTC"),
        (table(ROWS, preamble=[line.replace("UTC", "Europe/London") for line in PREAMBLE]), "UTC"),
        (b"# no header at all\r\n" * 80, "no header"),
        (table(ROWS, columns=(POWER, POWER)), "repeats"),
        (table(ROWS, columns=("Lost Production Total (kWh)", "Capacity factor")), "no signal"),
        (table([("yesterday", "1.0", "40.0")]), "conversion error to timestamp"),
        (table([("2016-02-30 00:00:00", "1.0", "40.0")]), "conversion error to timestamp"),
        (table([("2016-03-01 00:00:00", "fast", "40.0")]), "conversion error to double"),
        (table(ROWS).replace("°".encode(), b"\xb0"), "line 10 is not UTF-8"),
        (table(ROWS, columns=("Power (MW)", OIL)), "states MW where Power is read in kW"),
        (
            table(ROWS, columns=(POWER, "Power factor (cosphi) (%)")),
            "states % where Power factor",
        ),
        (table([("2016-03-01 00:00:00", "inf", "40.0")]), "non-finite value"),
        (
            table(ROWS, columns=("Power (kW)", "power (kW)")),
            "holds Power in two columns",
        ),
        (table([("", "1.0", "40.0")]), "no instant"),
        (table([("2015-12-31 23:50:00", "1.0", "40.0")]), "precedes 2016-01-01"),
        (table([("2016-03-01 00:00:00", "NaN", "NaN")]), "no reading"),
        (table([]), "no reading"),
    ],
)
def test_rejects_malformed_tables(tmp_path: Path, content: bytes, reason: str) -> None:
    root = farm_of(tmp_path, {member("1"): content})

    with pytest.raises(MalformedCorpusDataError, match=reason):
        reader_of(root, "1").describe()


@pytest.mark.parametrize("key", ["1/2016-02", "2/2016-03", "9/2016-03", "2016-03"])
def test_a_month_without_a_reading_or_outside_the_selection_is_refused(
    tmp_path: Path, key: str
) -> None:
    root = farm_of(tmp_path, {member("1"): table(ROWS), member("2"): table(ROWS)})

    with pytest.raises(UnknownUnitError):
        reader_of(root, "1").read_observations(UnitKey(key))


def three_months(root: Path) -> Path:
    """A turbine with two readings in January, two in February and one in March."""
    rows = [
        ("2016-01-10 00:00:00", "1.0", "40.0"),
        ("2016-02-10 00:00:00", "1.0", "40.0"),
        ("2016-03-10 00:00:00", "1.0", "NaN"),
    ]
    return farm_of(root, {member("1"): table(rows)})


def test_months_excluded_by_name_are_out_of_the_units_and_the_counts(tmp_path: Path) -> None:
    reader = GreenbyteCorpusReader(three_months(tmp_path), FARM, ("1",), excluded=("1/2016-02",))

    content = reader.describe().content

    assert [str(unit.key) for unit in reader.read_units()] == ["1/2016-01", "1/2016-03"]
    assert (content.unit_count, content.observation_count) == (2, 3)
    with pytest.raises(UnknownUnitError, match="excluded"):
        reader.read_observations(UnitKey("1/2016-02"))


def test_a_corpus_cut_by_months_is_another_version_and_one_cut_by_none_is_the_same(
    tmp_path: Path,
) -> None:
    root = three_months(tmp_path)
    whole = reader_of(root, "1").describe().content.checksum

    def cut(*months: str) -> Checksum:
        return GreenbyteCorpusReader(root, FARM, ("1",), months).describe().content.checksum

    assert cut() == whole
    assert cut("1/2016-02") not in (whole, cut("1/2016-03"))


def test_an_excluded_month_of_a_turbine_not_selected_is_refused() -> None:
    with pytest.raises(ValueError, match="within a selected turbine"):
        GreenbyteCorpusReader(SAMPLE, FARM, ("1",), excluded=("2/2016-06",))


def test_a_whole_turbine_is_read_through_the_cumulative_years() -> None:
    root = raw_root("kelmarsh")
    if root is None:
        pytest.skip("the Kelmarsh wind farm data is not downloaded")
    reader = GreenbyteCorpusReader(root, KELMARSH, ("1",))

    content = reader.describe().content
    units = {str(unit.key): unit for unit in reader.read_units()}

    # Turbine 1 sent its first reading on 23 January 2016 and reads in every month to the end
    # of 2024; its tables of 2023 and 2024 repeat every instant up to 81 times.
    assert (content.unit_count, content.observation_count) == (108, 26_175_969)
    assert units["1/2016-01"].extent.start == 22 * 24 + 12.0
    december = [o for o in reader.read_observations(UnitKey("1/2024-12")) if o.channel == "Power"]
    assert len(december) == len({o.time for o in december}) > 4_000
