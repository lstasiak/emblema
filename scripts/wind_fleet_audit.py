r"""What three open sources of wind turbine telemetry hold, measured from their files.

Two farms of one export (Kelmarsh, Penmanshiel) and a benchmark of yearly excerpts with labelled
faults (CARE to Compare) are read straight from the publishers' archives. A farm's columns are
chosen as its reader chooses them, by the reader's own signals; how the rows of an instant become
one reading, and every count, is this script's, so that the reader can be held against it.

    uv run scripts/wind_fleet_audit.py farm --farm Kelmarsh --root data/raw/kelmarsh \
        --out DIR [--turbines 1 3 4 5] [--before 2023-01-01]
    uv run scripts/wind_fleet_audit.py care --root data/raw/care/CARE_To_Compare --out DIR
    uv run scripts/wind_fleet_audit.py held --farm Kelmarsh --count 2 [--seed 1]

``farm`` reads every turbine table of a farm's archives and writes, on everything,
``<farm>-tables.csv`` (per table: its columns and unpacked bytes, its rows, its distinct instants,
the most rows one instant stands on, the instants whose rows disagree), ``<farm>-turbines.csv``
(per turbine: its instants and readings, its first and last reading, the first of a temperature,
its share of ten-minute steps) and ``<farm>-coverage.csv`` (per signal and year, the share of the
turbines' instants that hold it). On the turbines named and the instants before ``--before``
alone, so that what is held unread stays out of every number a publication is chosen by, it
writes ``<farm>-values.csv`` (per signal: its range, its median and deviations, and the share of
its readings beyond the bound of the robust score a publication would give it),
``<farm>-implausible.csv`` (per signal of a unit with a wide physical range stated here: the
readings outside it, the value most of them hold, and the hours they touch), ``<farm>-hourly.csv``
(per signal: how its hourly mean moves with the same signal of the farm's other turbines in the
same hour, and with itself an hour and a day before) and ``<farm>-independent.csv`` (per turbine,
how many signals its temperatures behave as). ``care`` writes ``care-excerpts.csv`` (per excerpt:
its turbine, lengths and event), ``care-groups.csv`` (the excerpts of a farm that share real
time, found by their wind) and ``care-sensors.csv``. ``held`` prints the turbines a seed holds
unread. The numbers describe the data and belong in no result.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.greenbyte import (  # noqa: E402
    GreenbyteCorpusReader,
    Signal,
)
from emblema.catalog.domain.channels.channel_statistics import MAD_TO_DEVIATION  # noqa: E402
from emblema.shared.kernel.ordering import seeded_rank  # noqa: E402

# pandas and pyarrow ship no type information, so a frame is whatever they hand back.
Frame = Any
Floats = NDArray[np.float64]
SIGNALS = GreenbyteCorpusReader.SIGNALS
TIME = "# Date and time"
TABLE = re.compile(r"Turbine_Data_(?P<farm>[^_]+)_(?P<turbine>[^_]+)_\d{4}-\d{2}-\d{2}_-_.*\.csv")
# The robust deviations the telemetry's publications hold a score to.
BOUND = 100.0
# Two excerpts share real time where their wind speeds at equal stamps move together this far.
SHARED_WIND = 0.7
# The fewest stamps two excerpts must share before their wind is compared.
LEAST_SHARED = 2000
# How the benchmark's descriptions spell degrees Celsius; two farms' files hold the replacement
# character where the degree sign was.
CELSIUS = ("Celsius", "°C", "\ufffdC")
ROWS_PER_DAY = 144
# A range no working sensor of the unit leaves on a turbine in Britain: wide enough that every
# reading outside it is a failed sensor's, not weather or load.
PHYSICAL_RANGES: Mapping[str, tuple[float, float]] = {
    "°C": (-50.0, 150.0),
    "m/s": (0.0, 60.0),
    "RPM": (-10.0, 2500.0),
}


@dataclass(frozen=True)
class TableFacts:
    """How one turbine table repeats its instants."""

    archive: str
    table: str
    turbine: str
    columns: int
    unpacked_bytes: int
    rows: int
    instants: int
    most_rows_of_an_instant: int
    instants_that_disagree: int


def columns_of(header: Sequence[str]) -> dict[str, str]:
    """Each header column that holds a signal's averages, by the signal it holds."""
    return {
        column: signal.name for signal in SIGNALS for column in header[1:] if signal.holds(column)
    }


def one_per_instant(frame: Frame) -> tuple[Frame, int]:
    """Rows that hold a reading as one row per instant, and the instants in dispute.

    A cumulative export repeats every instant with values in one of its rows; the value kept is
    the first one a column holds, and an instant is in dispute where a column holds two values.
    """
    counted = frame.groupby(level=0, sort=True)
    disputed = int((counted.nunique() > 1).any(axis=1).sum()) if len(frame) else 0
    return counted.first(), disputed


def most_rows_of_an_instant(stamps: NDArray[np.datetime64]) -> int:
    """The most rows any one instant stands on, among every row of a table."""
    return int(np.unique(stamps, return_counts=True)[1].max()) if len(stamps) else 0


def read_table(archive: zipfile.ZipFile, member: str) -> tuple[Frame, NDArray[np.datetime64], int]:
    """A table's rows that hold a reading, by signal, its every row's stamp, and its columns."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.csv as pc

    with archive.open(member) as stream:
        skipped = 0
        while not (line := stream.readline().decode("utf-8-sig")).startswith(TIME):
            skipped += 1
        header = next(csv.reader([line.rstrip("\r\n")]))
    named = columns_of(header)
    stamps, parts = [], []
    with archive.open(member) as stream:
        for batch in pc.open_csv(
            stream,
            read_options=pc.ReadOptions(skip_rows=skipped),
            convert_options=pc.ConvertOptions(
                include_columns=[TIME, *named],
                column_types={TIME: pa.timestamp("s"), **dict.fromkeys(named, pa.float64())},
                null_values=["", "NaN"],
            ),
        ):
            part = batch.to_pandas().set_index(TIME)
            stamps.append(part.index.to_numpy())
            parts.append(part[part.notna().any(axis=1)])
    every = np.concatenate(stamps) if stamps else np.zeros(0, dtype="datetime64[s]")
    frame = pd.concat(parts) if parts else pd.DataFrame()
    # Two spellings of one signal's unit are two columns of one signal.
    frame = frame.T.groupby(by=lambda column: named[column]).first().T
    return frame, every, len(header)


def read_farm(root: Path, farm: str) -> tuple[dict[str, Frame], list[TableFacts]]:
    """Every turbine of a farm as one frame of instants by signals, and each table's facts."""
    import pandas as pd

    frames: dict[str, list[Frame]] = {}
    facts: list[TableFacts] = []
    for path in sorted(root.glob(f"{farm}_SCADA_*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                match = TABLE.fullmatch(Path(member).name)
                if match is None:
                    continue
                frame, stamps, columns = read_table(archive, member)
                kept, disputed = one_per_instant(frame)
                frames.setdefault(match["turbine"], []).append(kept)
                facts.append(
                    TableFacts(
                        archive=path.name,
                        table=member,
                        turbine=match["turbine"],
                        columns=columns,
                        unpacked_bytes=archive.getinfo(member).file_size,
                        rows=len(stamps),
                        instants=len(np.unique(stamps)),
                        most_rows_of_an_instant=most_rows_of_an_instant(stamps),
                        instants_that_disagree=disputed,
                    )
                )
                print(f"{member}: {len(stamps):,} rows", file=sys.stderr)
    joined = {}
    for turbine, tables in sorted(frames.items()):
        whole = pd.concat(tables).sort_index()
        joined[turbine], disputed = one_per_instant(whole)
        if disputed:
            raise SystemExit(f"turbine {turbine}: {disputed} instants disagree between tables")
    return joined, facts


def coverage_rows(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    """Per signal and year, the share of the turbines' instants that hold a reading of it."""
    held: dict[tuple[str, int], int] = {}
    instants: dict[int, int] = {}
    for frame in frames.values():
        years = frame.index.year
        for year, count in zip(*np.unique(years, return_counts=True), strict=True):
            instants[int(year)] = instants.get(int(year), 0) + int(count)
        for signal, per_year in frame.notna().groupby(years).sum().items():
            for year, count in per_year.items():
                held[(str(signal), int(year))] = held.get((str(signal), int(year)), 0) + int(count)
    return [
        {"signal": signal, "year": year, "share": round(count / instants[year], 4)}
        for (signal, year), count in sorted(held.items())
    ]


def value_row(signal: Signal, values: Floats) -> dict[str, object]:
    """A signal's range and the share of its readings beyond the robust score's bound.

    The score is a reading's distance from the median over 1.4826 times the median absolute
    deviation, and over the standard deviation where that is nought, as a publication scores it.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    deviation = float(values.std())
    scale = MAD_TO_DEVIATION * mad if mad > 0.0 else deviation
    scores = np.abs(values - median) / scale if scale > 0.0 else np.zeros(len(values))
    return {
        "signal": signal.name,
        "unit": signal.unit or "",
        "readings": len(values),
        "minimum": float(values.min()),
        "median": median,
        "maximum": float(values.max()),
        "deviation": deviation,
        "robust_deviation": MAD_TO_DEVIATION * mad,
        f"share_beyond_{BOUND:g}": float((scores > BOUND).mean()),
    }


def value_rows(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    rows = []
    for signal in SIGNALS:
        values = np.concatenate(
            [frame[signal.name].dropna().to_numpy(np.float64) for frame in frames.values()]
        )
        if len(values):
            rows.append(value_row(signal, values))
    return rows


def implausible_rows(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    """Per signal with a physical range, its readings outside it and the hours they touch."""
    rows: list[dict[str, object]] = []
    for signal in SIGNALS:
        if signal.unit is None or signal.unit not in PHYSICAL_RANGES:
            continue
        low, high = PHYSICAL_RANGES[signal.unit]
        readings = hours = touched = 0
        outside: list[Floats] = []
        for frame in frames.values():
            values = frame[signal.name].dropna()
            beyond = values[(values < low) | (values > high)]
            readings += len(values)
            hours += int(values.index.floor("1h").nunique())
            touched += int(beyond.index.floor("1h").nunique())
            outside.append(beyond.to_numpy(np.float64))
        found = np.concatenate(outside) if outside else np.zeros(0)
        held, counts = np.unique(found.round(1), return_counts=True)
        rows.append(
            {
                "signal": signal.name,
                "unit": signal.unit,
                "readings": readings,
                "below": int((found < low).sum()),
                "above": int((found > high).sum()),
                "commonest_value": float(held[counts.argmax()]) if len(found) else "",
                "readings_at_it": int(counts.max()) if len(found) else 0,
                "hours": hours,
                "hours_touched": touched,
            }
        )
    return rows


def pairwise(columns: Frame) -> Floats:
    """The correlations between every two columns, each over the rows both hold."""
    matrix = columns.corr(min_periods=LEAST_SHARED).to_numpy()
    return np.asarray(matrix[np.triu_indices_from(matrix, 1)], dtype=np.float64)


def hourly_rows(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    """Per signal, how its hourly mean moves with the other turbines' and with its own past."""
    import pandas as pd

    hours = {turbine: frame.resample("1h").mean() for turbine, frame in frames.items()}
    rows: list[dict[str, object]] = []
    for signal in SIGNALS:
        levels: Frame = pd.DataFrame(
            {turbine: hourly[signal.name] for turbine, hourly in hours.items()}
        )
        changes = levels.diff()
        between = pairwise(changes)
        between = between[np.isfinite(between)]
        rows.append(
            {
                "signal": signal.name,
                "unit": signal.unit or "",
                "turbines": int(levels.notna().any().sum()),
                "changes_between_turbines_median": _median(between),
                "level_an_hour_before": _median([levels[t].autocorr(1) for t in levels]),
                "change_a_day_before": _median([changes[t].autocorr(24) for t in changes]),
            }
        )
    return rows


def _median(values: Iterable[float]) -> float | str:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else ""


def independent_signals(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    """Per turbine, how many signals the hourly changes of its temperatures behave as.

    The participation ratio of the eigenvalues of their correlation: as many as there are
    signals where none moves with another, one where all move together.
    """
    names = [signal.name for signal in SIGNALS if signal.unit == "°C"]
    rows: list[dict[str, object]] = []
    for turbine, frame in frames.items():
        changes = frame[names].resample("1h").mean().diff()
        read = [name for name in names if changes[name].notna().sum() >= LEAST_SHARED]
        values = np.linalg.eigvalsh(changes[read].corr().fillna(0.0).to_numpy())
        ratio = float(values.sum() ** 2 / (values**2).sum())
        rows.append({"turbine": turbine, "temperatures": len(read), "behave_as": round(ratio, 2)})
    return rows


def turbine_rows(frames: Mapping[str, Frame]) -> list[dict[str, object]]:
    """Per turbine, its span, its readings and how regular its instants are."""
    temperatures = [signal.name for signal in SIGNALS if signal.unit == "°C"]
    rows: list[dict[str, object]] = []
    for turbine, frame in frames.items():
        steps = np.diff(frame.index.to_numpy()).astype("timedelta64[s]").astype(np.int64)
        warm = frame[temperatures].notna().any(axis=1).to_numpy()
        rows.append(
            {
                "turbine": turbine,
                "instants": len(frame),
                "readings": int(frame.notna().sum().sum()),
                "first_reading": str(frame.index[0]),
                "first_temperature": str(frame.index[warm.argmax()]) if warm.any() else "",
                "last_reading": str(frame.index[-1]),
                "steps_of_ten_minutes": round(float((steps == 600).mean()), 5),
            }
        )
    return rows


def held_out(farm: str, turbines: Sequence[str], count: int, seed: int) -> list[str]:
    """The turbines a seed holds unread: the first ``count`` of the farm in the seed's order."""
    ranked = sorted(turbines, key=lambda turbine: seeded_rank(seed, farm, turbine))
    return sorted(ranked[:count])


# --- the benchmark of labelled faults --------------------------------------------------------


@dataclass(frozen=True)
class Excerpt:
    """One excerpt of the benchmark: a turbine's year and the window a detector is scored on."""

    farm: str
    event: int
    turbine: int
    label: str
    start: str
    training_days: float
    prediction_days: float
    event_days: float


@dataclass(frozen=True)
class SharedTime:
    """Which excerpts of a farm share real time.

    Attributes:
        groups: The excerpts that share it, largest group first.
        same_rows: Pairs of one turbine's excerpts that hold the same rows.
        other_rows: Pairs of one turbine's excerpts that hold other rows at equal stamps.
        least_within: The lowest wind correlation between two turbines taken to share time.
        most_between: The highest between two taken not to.
    """

    groups: list[list[int]]
    same_rows: int
    other_rows: int
    least_within: float
    most_between: float


def shared_time_groups(wind: Mapping[int, Frame], turbines: Mapping[int, int]) -> SharedTime:
    """The excerpts that share real time, and how same-turbine pairs overlap.

    Each excerpt's instants were shifted by whole years, so equal stamps are the same instant
    only by chance of the year. Two excerpts of different turbines share real time where their
    wind speeds at equal stamps move together; two of one turbine where their rows are the
    same.
    """
    import pandas as pd

    events = sorted(wind)
    table = pd.DataFrame({event: wind[event] for event in events})
    related = table.corr(min_periods=LEAST_SHARED)
    parent = {event: event for event in events}

    def root(event: int) -> int:
        while parent[event] != event:
            event = parent[event]
        return event

    same_rows = other_rows = 0
    within, between = [1.0], [-1.0]
    for index, one in enumerate(events):
        for other in events[index + 1 :]:
            moved = float(related.at[one, other])
            if not np.isfinite(moved):
                continue
            if turbines[one] == turbines[other]:
                both = table[[one, other]].dropna()
                shared = bool((both[one] == both[other]).mean() > 0.9)
                same_rows, other_rows = same_rows + shared, other_rows + (not shared)
            else:
                shared = moved > SHARED_WIND
                (within if shared else between).append(moved)
            if shared:
                parent[root(one)] = root(other)
    groups: dict[int, list[int]] = {}
    for event in events:
        groups.setdefault(root(event), []).append(event)
    return SharedTime(
        groups=sorted(groups.values(), key=lambda group: (-len(group), group)),
        same_rows=same_rows,
        other_rows=other_rows,
        least_within=min(within),
        most_between=max(between),
    )


def read_care(
    root: Path,
) -> tuple[list[Excerpt], list[dict[str, object]], list[dict[str, object]]]:  # pragma: no cover
    """Every excerpt's facts, each farm's groups of shared time, and its count of sensors."""
    import pandas as pd

    excerpts: list[Excerpt] = []
    groups: list[dict[str, object]] = []
    sensors: list[dict[str, object]] = []
    for farm in "ABC":
        folder = root / f"Wind Farm {farm}"
        events: Frame = pd.read_csv(folder / "event_info.csv", sep=";")
        described: Frame = pd.read_csv(
            folder / "feature_description.csv", sep=";", encoding_errors="replace"
        )
        sensors.append(
            {
                "farm": farm,
                "sensors": len(described),
                "temperatures": int(described.unit.isin(CELSIUS).sum()),
            }
        )
        header = pd.read_csv(next((folder / "datasets").glob("*.csv")), sep=";", nrows=0).columns
        speed = next(c for c in header if c.startswith("wind_speed") and c.endswith("_avg"))
        wind: dict[int, Frame] = {}
        turbines: dict[int, int] = {}
        for event in events.to_dict("records"):
            number = int(event["event_id"])
            frame: Frame = pd.read_csv(
                folder / "datasets" / f"{number}.csv",
                sep=";",
                usecols=["time_stamp", "asset_id", "train_test", speed],
                parse_dates=["time_stamp"],
            )
            training = frame[frame.train_test == "train"]
            prediction = frame[frame.train_test != "train"]
            starts: Any = pd.Timestamp(event["event_start"])
            ends: Any = pd.Timestamp(event["event_end"])
            day = pd.Timedelta("1D")
            turbines[number] = int(frame.asset_id.iloc[0])
            wind[number] = frame.groupby("time_stamp")[speed].first().astype(float)
            excerpts.append(
                Excerpt(
                    farm=farm,
                    event=number,
                    turbine=turbines[number],
                    label=str(event["event_label"]),
                    start=str(frame.time_stamp.min()),
                    training_days=round(len(training) / ROWS_PER_DAY, 1),
                    prediction_days=round(len(prediction) / ROWS_PER_DAY, 1),
                    event_days=round((ends - starts) / day, 2),
                )
            )
            print(f"farm {farm}, event {number}: {len(frame):,} rows", file=sys.stderr)
        shared = shared_time_groups(wind, turbines)
        for number, group in enumerate(shared.groups, 1):
            groups.append(
                {
                    "farm": farm,
                    "group": number,
                    "excerpts": len(group),
                    "turbines": len({turbines[event] for event in group}),
                    "events": " ".join(map(str, group)),
                    "pairs_of_one_turbine_with_the_same_rows": shared.same_rows,
                    "pairs_of_one_turbine_with_other_rows": shared.other_rows,
                    "least_wind_correlation_within_groups": round(shared.least_within, 3),
                    "most_wind_correlation_between_groups": round(shared.most_between, 3),
                }
            )
    return excerpts, groups, sensors


def write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def farm_named(name: str) -> tuple[str, ...]:
    known = {
        farm.name: farm.turbines
        for farm in (GreenbyteCorpusReader.KELMARSH, GreenbyteCorpusReader.PENMANSHIEL)
    }
    if name not in known:
        raise SystemExit(f"unknown farm {name!r}; expected one of {sorted(known)}")
    return known[name]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the archives
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    farming = commands.add_parser("farm", help="a farm's tables, coverage, values and neighbours")
    farming.add_argument("--farm", required=True)
    farming.add_argument("--root", type=Path, required=True)
    farming.add_argument("--out", type=Path, required=True)
    farming.add_argument("--turbines", nargs="*", help="the turbines the values are read on")
    farming.add_argument("--before", help="the instant the values are read up to, exclusive")
    caring = commands.add_parser("care", help="the benchmark's excerpts and their shared time")
    caring.add_argument("--root", type=Path, required=True)
    caring.add_argument("--out", type=Path, required=True)
    holding = commands.add_parser("held", help="the turbines a seed holds unread")
    holding.add_argument("--farm", required=True)
    holding.add_argument("--count", type=int, required=True)
    holding.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)

    if arguments.command == "held":
        turbines = farm_named(arguments.farm)
        print(" ".join(held_out(arguments.farm, turbines, arguments.count, arguments.seed)))
        return
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.command == "care":
        excerpts, groups, sensors = read_care(arguments.root)
        write(arguments.out / "care-excerpts.csv", [asdict(excerpt) for excerpt in excerpts])
        write(arguments.out / "care-groups.csv", groups)
        write(arguments.out / "care-sensors.csv", sensors)
        return
    farm_named(arguments.farm)
    frames, facts = read_farm(arguments.root, arguments.farm)
    label = arguments.farm.lower()
    write(arguments.out / f"{label}-tables.csv", [asdict(fact) for fact in facts])
    write(arguments.out / f"{label}-turbines.csv", turbine_rows(frames))
    write(arguments.out / f"{label}-coverage.csv", coverage_rows(frames))
    unknown = sorted(set(arguments.turbines or ()) - set(frames))
    if unknown:
        raise SystemExit(f"{arguments.farm} has no turbines {unknown}")
    end = pd.Timestamp.max if arguments.before is None else pd.Timestamp(arguments.before)
    chosen = {
        turbine: frame[frame.index < end]
        for turbine, frame in frames.items()
        if not arguments.turbines or turbine in arguments.turbines
    }
    write(arguments.out / f"{label}-values.csv", value_rows(chosen))
    write(arguments.out / f"{label}-implausible.csv", implausible_rows(chosen))
    write(arguments.out / f"{label}-hourly.csv", hourly_rows(chosen))
    write(arguments.out / f"{label}-independent.csv", independent_signals(chosen))
    print(f"{arguments.farm}: {len(frames)} turbines, {len(facts)} tables")


if __name__ == "__main__":  # pragma: no cover
    main()
