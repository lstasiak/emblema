"""The portal's results become readings through a location's main organisation, at instants in UTC.

Held to hand-built result rows whose readings, instants and summary are known; fetching from the
portal is not tested here.
"""

from pathlib import Path

import pytest

from scripts.wqp_result_table import instant_of, numeric, ranked, readings_of, write

EPOCH_HOURS_2020 = 438_288.0  # 2020-01-01T00:00Z in hours since the epoch


def row(
    organisation: str, name: str, value: str, clock: str = "10:00:00", zone: str = "EST"
) -> dict[str, str]:
    return {
        "OrganizationIdentifier": organisation,
        "CharacteristicName": name,
        "ResultMeasureValue": value,
        "ActivityStartDate": "2020-01-01",
        "ActivityStartTime/Time": clock,
        "ActivityStartTime/TimeZoneCode": zone,
    }


def test_a_zoned_time_is_moved_to_utc() -> None:
    stamp = instant_of("2020-01-01", "10:00:00", "EST")

    assert stamp.hours == EPOCH_HOURS_2020 + 15.0
    assert not stamp.dated_only
    assert not stamp.unzoned


def test_a_date_without_a_time_is_its_midnight_and_a_time_without_a_known_zone_is_utc() -> None:
    dated = instant_of("2020-01-01", "", "")
    unknown = instant_of("2020-01-01", "10:30", "XYZ")

    assert (dated.hours, dated.dated_only, dated.unzoned) == (EPOCH_HOURS_2020, True, False)
    assert (unknown.hours, unknown.dated_only, unknown.unzoned) == (
        EPOCH_HOURS_2020 + 10.5,
        False,
        True,
    )


def test_only_a_finite_number_is_a_value() -> None:
    assert numeric("7.5") == 7.5
    assert numeric("") is None
    assert numeric("<0.01") is None
    assert numeric("nan") is None


def test_a_location_is_read_through_the_organisation_that_reports_most() -> None:
    rows = [
        row("A", "Nitrate", "1.0"),
        row("A", "Oxygen", "7.0", clock=""),
        row("A", "pH", "not detected"),
        row("B", "Nitrate as N", "1.0"),
    ]

    summary, readings = readings_of("site", rows, least=2)

    assert readings == [
        ("site", "Nitrate", EPOCH_HOURS_2020 + 15.0),
        ("site", "Oxygen", EPOCH_HOURS_2020 + 5.0),
    ]
    assert (summary.organisation, summary.results, summary.readings) == ("A", 4, 2)
    assert summary.organisation_share == 0.75
    assert (summary.dated_only, summary.unzoned, summary.kept) == (1, 0, True)


def test_a_location_with_too_few_readings_is_fetched_but_not_kept() -> None:
    summary, readings = readings_of("site", [row("A", "Nitrate", "1.0")], least=2)

    assert readings == []
    assert not summary.kept
    assert readings_of("empty", [], least=2)[0].results == 0


def test_locations_are_taken_in_the_seeded_order_of_their_identifiers(tmp_path: Path) -> None:
    listing = tmp_path / "station.csv"
    listing.write_text("MonitoringLocationIdentifier,Other\nb,1\na,2\nc,3\na,4\n")

    order = ranked(listing, seed=1)

    assert sorted(order) == ["a", "b", "c"]
    assert order == ranked(listing, seed=1)
    broken = tmp_path / "broken.csv"
    broken.write_text("Name\nx\n")
    with pytest.raises(KeyError):
        ranked(broken, seed=1)


def test_rows_are_written_under_their_header(tmp_path: Path) -> None:
    write(tmp_path / "readings.csv", ("unit", "channel", "time"), [("site", "pH", "1.5")])

    assert (tmp_path / "readings.csv").read_text().splitlines() == [
        "unit,channel,time",
        "site,pH,1.5",
    ]


def test_a_result_dated_alone_is_no_reading_when_only_timed_results_are_read() -> None:
    rows = [row("A", "Nitrate", "1.0"), row("A", "Oxygen", "7.0", clock="")]

    summary, readings = readings_of("site", rows, least=1, timed_only=True)

    assert [name for _, name, _ in readings] == ["Nitrate"]
    assert (summary.readings, summary.dated_only) == (1, 1)
