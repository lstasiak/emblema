"""Turn the Water Quality Portal results of drawn locations into the long table the raw audit reads.

The frame is the portal's Station listing for stream, lake and reservoir locations with at least 100
results over the years read. Locations are taken in the order the project's seeded ranking gives
their identifiers, their results fetched one location at a time, and a location is kept when it
holds enough readings, until enough are kept: the same units the raw audit's draw would take from
the whole frame, without fetching it whole.

    uv run scripts/wqp_result_table.py --stations data/raw/wqp/station.csv \
        --cache data/raw/wqp/results --out data/report/raw-regime-audit/wqp
    uv run scripts/raw_regime_audit.py --name wqp --root data/report/raw-regime-audit/wqp \
        --pattern readings.csv --unit unit --channel channel --time time

A location is read through the organisation that reports most of its results, since one quantity
reported by two organisations under two names would read as two channels. A reading is a result
with a numeric ``ResultMeasureValue``; its channel is the ``CharacteristicName`` as published, its
instant the activity's start date and time in hours since the epoch, UTC, a date without a time at
its midnight and a time without a known zone as UTC; with ``--timed-only`` a result dated alone is
no reading. Writes ``readings.csv`` (unit, channel, time)
and ``frame.csv`` (one row per location fetched: its results, the share the organisation kept
reports, its readings, and the readings without a time or a known zone).
"""

import argparse
import csv
import io
import math
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.kernel.ordering import seeded_rank

RESULTS = "https://www.waterqualitydata.us/data/Result/search"
FIRST, LAST = "01-01-2010", "12-31-2023"
SECONDS_PER_HOUR = 3600.0
# The offsets from UTC, in hours, of the zone codes the portal's activities carry.
ZONES: Mapping[str, float] = {
    "UTC": 0.0,
    "GMT": 0.0,
    "AST": -4.0,
    "ADT": -3.0,
    "EST": -5.0,
    "EDT": -4.0,
    "CST": -6.0,
    "CDT": -5.0,
    "MST": -7.0,
    "MDT": -6.0,
    "PST": -8.0,
    "PDT": -7.0,
    "AKST": -9.0,
    "AKDT": -8.0,
    "HST": -10.0,
    "HAST": -10.0,
    "HADT": -9.0,
    "SST": -11.0,
    "GST": 10.0,
    "CHST": 10.0,
}


@dataclass(frozen=True)
class Instant:
    """When a result was taken, and whether its stamp was whole.

    Attributes:
        hours: Hours since the epoch, UTC.
        dated_only: The activity gave a date and no time.
        unzoned: The activity gave a time and no zone the table knows.
    """

    hours: float
    dated_only: bool
    unzoned: bool


def instant_of(day: str, clock: str, zone: str) -> Instant:
    taken = date.fromisoformat(day)
    moment = datetime(taken.year, taken.month, taken.day, tzinfo=UTC)
    if clock:
        parts = [int(part) for part in clock.split(":")]
        moment += timedelta(
            hours=parts[0], minutes=parts[1], seconds=parts[2] if len(parts) > 2 else 0
        )
    offset = ZONES.get(zone.strip().upper())
    if offset is not None:
        moment -= timedelta(hours=offset)
    return Instant(
        hours=moment.timestamp() / SECONDS_PER_HOUR,
        dated_only=not clock,
        unzoned=bool(clock) and offset is None,
    )


def numeric(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class LocationRow:
    """One location fetched; a plain record written to ``frame.csv``."""

    location: str
    results: int
    organisation: str
    organisation_share: float
    readings: int
    dated_only: int
    unzoned: int
    kept: bool


def readings_of(
    location: str, rows: Sequence[Mapping[str, str]], least: int, timed_only: bool = False
) -> tuple[LocationRow, list[tuple[str, str, float]]]:
    """A location's readings through its main organisation, and what reading them left out."""
    if not rows:
        return LocationRow(location, 0, "", 0.0, 0, 0, 0, False), []
    counts = Counter(row["OrganizationIdentifier"] for row in rows)
    organisation, reported = counts.most_common(1)[0]
    readings, dated_only, unzoned = [], 0, 0
    for row in rows:
        if (
            row["OrganizationIdentifier"] != organisation
            or numeric(row["ResultMeasureValue"]) is None
        ):
            continue
        stamp = instant_of(
            row["ActivityStartDate"],
            row["ActivityStartTime/Time"],
            row["ActivityStartTime/TimeZoneCode"],
        )
        dated_only += stamp.dated_only
        if timed_only and stamp.dated_only:
            continue
        unzoned += stamp.unzoned
        readings.append((location, row["CharacteristicName"], stamp.hours))
    kept = len(readings) >= least
    summary = LocationRow(
        location=location,
        results=len(rows),
        organisation=organisation,
        organisation_share=reported / len(rows),
        readings=len(readings),
        dated_only=dated_only,
        unzoned=unzoned,
        kept=kept,
    )
    return summary, readings if kept else []


def ranked(stations: Path, seed: int) -> list[str]:
    """The listing's locations in the order the seeded ranking gives their identifiers."""
    with stations.open(newline="") as file:
        names = {row["MonitoringLocationIdentifier"] for row in csv.DictReader(file)}
    return sorted(names, key=lambda name: seeded_rank(seed, name))


def fetched(
    location: str, cache: Path, attempts: int = 4
) -> list[dict[str, str]]:  # pragma: no cover - the network
    """The location's results over the years read, from the cache or the portal."""
    path = cache / f"{urllib.parse.quote(location, safe='')}.zip"
    if not path.exists():
        query = urllib.parse.urlencode(
            {
                "siteid": location,
                "startDateLo": FIRST,
                "startDateHi": LAST,
                "mimeType": "csv",
                "zip": "yes",
            }
        )
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(f"{RESULTS}?{query}", timeout=600) as response:
                    content = response.read()
                break
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(30 * (attempt + 1))
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    with zipfile.ZipFile(path) as archive:
        [member] = archive.namelist()
        text = archive.read(member).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def write(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stations", type=Path, required=True, help="the Station listing, CSV")
    parser.add_argument("--cache", type=Path, required=True, help="where fetched results are kept")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--least", type=int, default=50, help="readings a location needs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timed-only", action="store_true", help="drop results dated alone")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    locations: list[LocationRow] = []
    readings: list[tuple[str, str, float]] = []
    for location in ranked(arguments.stations, arguments.seed):
        summary, held = readings_of(
            location, fetched(location, arguments.cache), arguments.least, arguments.timed_only
        )
        locations.append(summary)
        readings.extend(held)
        kept = sum(row.kept for row in locations)
        print(f"{location}: {summary.readings} readings, kept {kept}", flush=True)
        if kept == arguments.units:
            break
    write(
        arguments.out / "readings.csv",
        ("unit", "channel", "time"),
        ((u, c, repr(t)) for u, c, t in readings),
    )
    columns = [field.name for field in fields(LocationRow)]
    write(
        arguments.out / "frame.csv",
        columns,
        ([asdict(row)[c] for c in columns] for row in locations),
    )


if __name__ == "__main__":
    main()
