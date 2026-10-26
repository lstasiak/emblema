"""The Water Quality Portal reader keeps what is a reading and the channels measured most widely.

Each rule is held to a batch built by hand in the portal's narrow result layout, whose readings are
known; the miniature sample of real batches is read whole; the downloaded batches are read where
they were fetched.
"""

import csv
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.wqp import WaterQualityPortalCorpusReader
from emblema.catalog.domain.channels.channel_schema import Channel
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.sampling import SamplingRegime
from tests.support.corpora import raw_root, sample

COLUMNS = (
    "OrganizationIdentifier",
    "MonitoringLocationIdentifier",
    "ProviderName",
    "ActivityIdentifier",
    "ActivityMediaName",
    "ActivityTypeCode",
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "ActivityStartTime/TimeZoneCode",
    "ActivityDepthHeightMeasure/MeasureValue",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "StatisticalBaseCode",
    "ResultDepthHeightMeasure/MeasureValue",
)


def result(
    location: str = "SITE-1",
    name: str = "pH",
    value: str = "7.0",
    *,
    organisation: str = "ORG",
    activity: str | None = None,
    media: str = "Water",
    kind: str = "Sample-Routine",
    day: str = "2015-06-01",
    clock: str = "10:00:00",
    zone: str = "UTC",
    unit: str = "std units",
    provider: str = "STORET",
    fraction: str = "",
    statistic: str = "",
    depth: str = "",
) -> dict[str, str]:
    """A result row; each row is an activity of its own unless one is named."""
    return dict(
        zip(
            COLUMNS,
            (
                organisation,
                location,
                provider,
                activity or f"{location}-{day}-{clock}-{name}-{value}-{depth}-{fraction}",
                media,
                kind,
                day,
                clock,
                zone,
                "",
                name,
                fraction,
                value,
                unit,
                statistic,
                depth,
            ),
            strict=True,
        )
    )


def batch(
    root: Path, number: int, rows: list[dict[str, str]], columns: tuple[str, ...] = COLUMNS
) -> None:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "batches" / f"batch-{number:05d}.zip", "w") as archive:
        archive.writestr("result.csv", text.getvalue())


def hours(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp() / 3600.0


def observations(
    reader: WaterQualityPortalCorpusReader, key: str
) -> list[tuple[str, float, float]]:
    return [(o.channel, o.time, o.value) for o in reader.read_observations(UnitKey(key))]


def test_only_timed_numeric_water_readings_of_routine_samples_count(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="7.1"),
            result(value="7.2", media="Sediment"),
            result(value="7.3", kind="Quality Control Sample-Field Replicate"),
            result(value="7.4", clock=""),
            result(value="<0.5"),
            result(value="nan"),
            result(value="7.5", day="2009-12-31"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert observations(reader, "STORET/SITE-1") == [("pH", hours("2015-06-01T10:00"), 7.1)]


def test_a_zoned_time_is_read_in_utc_and_an_unknown_zone_as_utc(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [result(value="1", zone="EST"), result(value="2", clock="12:30", zone="XYZ")],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert observations(reader, "STORET/SITE-1") == [
        ("pH", hours("2015-06-01T12:30"), 2.0),
        ("pH", hours("2015-06-01T15:00"), 1.0),
    ]


def test_a_location_is_read_through_the_organisation_that_reports_most(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", organisation="BIG"),
            result(value="2", organisation="BIG", clock="11:00:00"),
            result(value="3", organisation="SMALL", clock="12:00:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [1.0, 2.0]


def test_a_characteristic_is_read_in_its_most_frequent_unit_whatever_its_case(
    tmp_path: Path,
) -> None:
    batch(
        tmp_path,
        0,
        [
            result(name="Nitrate", value="1", unit="mg/L"),
            result(name="Nitrate", value="2", unit="mg/L", clock="11:00"),
            result(name="Nitrate", value="3", unit="mg/l", clock="12:00"),
            result(name="Nitrate", value="4000", unit="ug/L", clock="13:00"),
            result(name="Nitrate", value="5000", unit="ug/L", clock="14:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert reader.describe().channel_schema.channels == frozenset({Channel("Nitrate", "mg/L")})
    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [1.0, 2.0, 3.0]


def test_the_channels_kept_are_those_measured_at_the_most_locations(tmp_path: Path) -> None:
    rows = [result(location=f"S{n}", name="Wide") for n in range(3)]
    rows += [result(location="S0", name="Dense", clock=f"{h:02d}:00") for h in range(10)]
    rows += [result(location=f"S{n}", name="Middle") for n in range(2)]
    batch(tmp_path, 0, rows)

    reader = WaterQualityPortalCorpusReader(tmp_path, channels=2, least=1)

    names = {channel.name for channel in reader.describe().channel_schema.channels}
    assert names == {"Wide", "Middle"}
    assert reader.describe().content.observation_count == 5


def test_a_location_enters_with_enough_readings_on_the_channels_kept(tmp_path: Path) -> None:
    rows = [result(location="RICH", clock=f"{h:02d}:00") for h in range(3)]
    rows += [result(location="POOR")]
    batch(tmp_path, 0, rows)

    reader = WaterQualityPortalCorpusReader(tmp_path, least=2)

    assert [str(unit.key) for unit in reader.read_units()] == ["STORET/RICH"]
    with pytest.raises(UnknownUnitError):
        reader.read_observations(UnitKey("STORET/POOR"))


def test_a_location_spans_the_years_fetched_and_its_key_may_hold_a_slash(tmp_path: Path) -> None:
    batch(tmp_path, 0, [result(location="A/B-1", provider="NWIS")])

    [unit] = WaterQualityPortalCorpusReader(tmp_path, least=1).read_units()

    assert str(unit.key) == "NWIS/A/B-1"
    assert unit.key.part == "NWIS"
    assert (unit.extent.start, unit.extent.end) == (
        hours("2010-01-01T00:00"),
        hours("2024-01-01T00:00"),
    )


def test_units_come_batch_by_batch(tmp_path: Path) -> None:
    batch(tmp_path, 0, [result(location="Z")])
    batch(tmp_path, 1, [result(location="A")])

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [str(unit.key) for unit in reader.read_units()] == ["STORET/Z", "STORET/A"]
    assert reader.describe().sampling_regime is SamplingRegime.IRREGULAR


def test_a_location_in_two_batches_is_refused(tmp_path: Path) -> None:
    batch(tmp_path, 0, [result()])
    batch(tmp_path, 1, [result()])

    with pytest.raises(MalformedCorpusDataError, match="already read"):
        WaterQualityPortalCorpusReader(tmp_path, least=1).describe()


@pytest.mark.parametrize(
    ("rows", "columns", "match"),
    [
        ([result()], COLUMNS[:-1], "no column"),
        ([result(day="2015-13-01")], COLUMNS, "not an instant"),
        ([result(clock="ten")], COLUMNS, "not a time of day"),
        ([result(clock="10:61")], COLUMNS, "not an instant"),
        ([result(location="")], COLUMNS, "non-blank"),
    ],
)
def test_a_batch_that_is_not_the_portals_is_refused(
    tmp_path: Path, rows: list[dict[str, str]], columns: tuple[str, ...], match: str
) -> None:
    batch(tmp_path, 0, rows, columns)

    with pytest.raises(MalformedCorpusDataError, match=match):
        WaterQualityPortalCorpusReader(tmp_path, least=1).describe()


def test_an_archive_that_is_not_one_zipped_table_is_refused(tmp_path: Path) -> None:
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "batch-00000.zip").write_bytes(b"not a zip")
    with pytest.raises(MalformedCorpusDataError, match="not a zip"):
        WaterQualityPortalCorpusReader(tmp_path).describe()
    with zipfile.ZipFile(tmp_path / "batches" / "batch-00000.zip", "w") as archive:
        archive.writestr("a.csv", "")
        archive.writestr("b.csv", "")
    with pytest.raises(MalformedCorpusDataError, match="one member"):
        WaterQualityPortalCorpusReader(tmp_path).describe()


def test_a_root_without_batches_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusDataNotFoundError, match="wqp_fetch"):
        WaterQualityPortalCorpusReader(tmp_path).describe()


def test_a_statistic_of_other_results_is_no_reading(tmp_path: Path) -> None:
    batch(
        tmp_path, 0, [result(value="7.0"), result(value="9.0", statistic="Maximum", clock="11:00")]
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [7.0]


def test_an_activity_repeating_a_characteristic_gives_its_results_no_instant(
    tmp_path: Path,
) -> None:
    logger = [result(value=str(v), activity="LOGGER") for v in (14.0, 20.0, 23.0)]
    batch(tmp_path, 0, [*logger, result(value="7.0", clock="12:00")])

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert observations(reader, "STORET/SITE-1") == [("pH", hours("2015-06-01T12:00"), 7.0)]


def test_one_reading_is_kept_per_channel_and_instant_whole_water_then_shallowest_then_first(
    tmp_path: Path,
) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", fraction="Dissolved"),
            result(value="2", depth="5"),
            result(value="3", depth="0.5"),
            result(value="4", depth="0.5"),
            result(value="5", clock="11:00", depth="2"),
            result(value="6", clock="11:00"),
            result(value="7", clock="12:00", depth="deep"),
            result(value="8", clock="12:00", depth="9", fraction="Total"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [3.0, 6.0, 8.0]


def test_ties_fall_to_the_first_name_and_code(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(name="Alpha", value="1", organisation="B"),
            result(name="Alpha", value="2", organisation="A", clock="11:00"),
            result(name="Beta", value="1", unit="b", clock="12:00", organisation="A"),
            result(name="Beta", value="2", unit="a", clock="13:00", organisation="A"),
            result(location="SITE-2", name="Gamma", organisation="A"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, channels=2, least=1)

    assert observations(reader, "STORET/SITE-1") == [
        ("Alpha", hours("2015-06-01T11:00"), 2.0),
        ("Beta", hours("2015-06-01T13:00"), 2.0),
    ]
    assert reader.describe().channel_schema.names == ("Alpha", "Beta")


def test_the_years_hold_their_first_instant_and_not_their_end(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", day="2010-01-01", clock="00:00"),
            result(value="2", day="2023-12-31", clock="23:59:59"),
            result(value="3", day="2023-12-31", clock="20:00", zone="est"),
            result(value="4", day="2024-01-01", clock="00:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [1.0, 2.0]


def test_describing_again_reads_the_batches_as_they_now_are(tmp_path: Path) -> None:
    batch(tmp_path, 0, [result()])
    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)
    reader.describe()

    batch(tmp_path, 0, [result(), result(location="SITE-2")])

    assert reader.describe().content.unit_count == 2


@pytest.mark.parametrize(("channels", "least"), [(0, 1), (1, 0)])
def test_a_reader_keeping_nothing_is_refused(tmp_path: Path, channels: int, least: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        WaterQualityPortalCorpusReader(tmp_path, channels=channels, least=least)


def test_a_single_measurement_basis_is_a_reading_and_a_computed_one_is_not(
    tmp_path: Path,
) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", statistic="MPN"),
            result(value="2", statistic="Instantaneous", clock="11:00"),
            result(value="3", statistic="Delta", clock="12:00"),
            result(value="4", statistic="mean", clock="13:00"),
            result(value="5", media=" water ", clock="14:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [1.0, 2.0, 5.0]


def test_an_activity_may_report_a_characteristic_once_per_fraction_unit_and_depth(
    tmp_path: Path,
) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", activity="A", depth="1"),
            result(value="2", activity="A", depth="3", clock="10:00"),
            result(value="3", activity="B", fraction="Dissolved", clock="11:00"),
            result(value="4", activity="B", fraction="unfiltered", clock="11:00"),
            result(value="5", activity="C", clock="12:00"),
            result(value="6", activity="C", unit="deg F", clock="12:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [1.0, 4.0, 5.0]


def test_the_main_organisation_is_counted_on_every_row_of_the_location(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(value="1", organisation="A"),
            result(value="2", organisation="B", clock="11:00"),
            result(value="3", organisation="B", media="Sediment", clock="12:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert [value for _, _, value in observations(reader, "STORET/SITE-1")] == [2.0]


def test_a_unit_is_spelt_as_most_readings_spell_it(tmp_path: Path) -> None:
    batch(
        tmp_path,
        0,
        [
            result(name="Nitrate", value="1", unit="mg/l"),
            result(name="Nitrate", value="2", unit="mg/L", clock="11:00"),
            result(name="Nitrate", value="3", unit="mg/L", clock="12:00"),
        ],
    )

    reader = WaterQualityPortalCorpusReader(tmp_path, least=1)

    assert reader.describe().channel_schema.channels == frozenset({Channel("Nitrate", "mg/L")})


def test_a_batch_saved_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    batch(tmp_path, 0, [result()])
    path = tmp_path / "batches" / "batch-00000.zip"
    with zipfile.ZipFile(path) as archive:
        text = archive.read("result.csv")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.csv", b"\xef\xbb\xbf" + text)

    assert WaterQualityPortalCorpusReader(tmp_path, least=1).describe().content.unit_count == 1


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (result(day="20150601"), "not an instant"),
        (result(day="0001-01-01", clock="00:00", zone="GST"), "not an instant"),
        (result(clock="\u0661\u0660:00"), "not a time of day"),
    ],
)
def test_a_stamp_the_portal_does_not_write_is_refused(
    tmp_path: Path, row: dict[str, str], match: str
) -> None:
    batch(tmp_path, 0, [row])

    with pytest.raises(MalformedCorpusDataError, match=match):
        WaterQualityPortalCorpusReader(tmp_path, least=1).describe()


def test_a_batch_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    (tmp_path / "batches").mkdir()
    with zipfile.ZipFile(tmp_path / "batches" / "batch-00000.zip", "w") as archive:
        archive.writestr("result.csv", ",".join(COLUMNS).encode() + b"\n\xff\xfe\n")

    with pytest.raises(MalformedCorpusDataError, match="UTF-8"):
        WaterQualityPortalCorpusReader(tmp_path, least=1).describe()


def test_the_sample_is_read_in_time_order_on_its_channels() -> None:
    reader = WaterQualityPortalCorpusReader(sample("wqp"))

    description = reader.describe()
    units = list(reader.read_units())

    assert [str(unit.key) for unit in units] == [
        "NWIS/USGS-01387610",
        "NWIS/USGS-423412082471600",
        "STORET/CHESAPEAKEMONITORINGCOOP-CMC.WSR.WASPRU002",
        "STORET/21VASWCB-1ALIV016.50",
        "STORET/21VASWCB-9-SNC008.04",
    ]
    assert (description.content.unit_count, description.content.observation_count) == (5, 310)
    assert len(description.channel_schema.channels) == 64
    for unit in units:
        read = list(reader.read_observations(unit.key))
        assert [o.time for o in read] == sorted(o.time for o in read)
        assert {o.channel for o in read} <= set(description.channel_schema.names)


@pytest.mark.skipif(raw_root("wqp") is None, reason="the raw corpus is not on this machine")
def test_the_downloaded_batches_keep_the_channels_and_locations_their_rule_names() -> None:
    root = raw_root("wqp")
    assert root is not None
    reader = WaterQualityPortalCorpusReader(root)

    description = reader.describe()

    assert len(description.channel_schema.channels) == WaterQualityPortalCorpusReader.CHANNELS
    assert description.content.observation_count >= description.content.unit_count * 50
