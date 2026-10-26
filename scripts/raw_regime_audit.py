"""Measure how a candidate source is sampled, from its raw files, before anything reads it.

A source enters a pretraining only through a reader, and a reader is worth writing only for a
source whose sampling is the one wanted. This audit answers the first question without the second:
it reads a long table straight from the publisher's files (a reading per row: its unit, its
channel, its instant) and measures each unit with the statistics the coverage report applies to
the corpora that have readers (``regime_coverage_report.unit_row``), so the two tables compare.

    uv run scripts/raw_regime_audit.py --name physionet2012 --root data/raw/physionet2012 \
        --pattern 'set-[ab]/*.txt' --unit-from-file --channel Parameter --time Time \
        --clock --value Value --missing -1 --skip RecordID Age Gender Height ICUType \
        --out data/report/raw-regime-audit

A unit is a file (``--unit-from-file``, keyed ``<directory>/<stem>`` as the readers key theirs) or
the value of a column (``--unit``). Units are drawn by seed before their readings are collected,
so a rerun reads the same units, and the draw over the same keys is the coverage report's. A unit
left with no reading is counted as empty and measured as none. Writes ``<name>.csv`` with the
coverage report's unit columns and prints the medians with their 10th and 90th percentiles. The
numbers describe the data and belong in no result.
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.regime_coverage_report import (
    UNIT_COLUMNS,
    Times,
    UnitRow,
    chosen_keys,
    quantiles,
    unit_rows_of,
    write,
)

_MINUTES_PER_HOUR = 60


@dataclass(frozen=True)
class LongTable:
    """Where a source's readings sit in its files, and which of them are no reading.

    Built from the command line; nothing reads it from outside the process. A unit key holding a
    slash is counted within the part before it, as the coverage report counts its units.

    Attributes:
        root: The directory the files lie under.
        pattern: The files, as a glob relative to ``root``.
        channel: The column naming a reading's channel.
        time: The column holding its instant.
        unit: The column naming its unit; ``None`` makes each file a unit.
        clock: Whether the instant is an ``HH:MM`` stamp, read in hours, rather than a number.
        value: The column holding its value, read only to drop the missing.
        missing: Values that mark a reading that was not taken.
        skip: Channels that are no reading at all (identifiers, descriptors).
        delimiter: The field separator.
    """

    root: Path
    pattern: str
    channel: str
    time: str
    unit: str | None = None
    clock: bool = False
    value: str | None = None
    missing: frozenset[str] = frozenset()
    skip: frozenset[str] = frozenset()
    delimiter: str = ","

    def files(self) -> list[Path]:
        return sorted(path for path in self.root.glob(self.pattern) if path.is_file())

    def key_of(self, path: Path) -> str:
        return f"{path.parent.name}/{path.stem}"

    def rows(self, path: Path) -> Iterator[dict[str, str]]:
        """The rows of one file.

        Raises:
            ValueError: If its header lacks a column the table reads.
        """
        wanted = [self.channel, self.time, *(c for c in (self.unit, self.value) if c is not None)]
        with path.open(newline="") as file:
            reader = csv.DictReader(file, delimiter=self.delimiter)
            absent = [column for column in wanted if column not in (reader.fieldnames or ())]
            if absent:
                raise ValueError(f"{path}: no column {', '.join(absent)} in the header")
            yield from reader

    def unit_of(self, path: Path, row: dict[str, str]) -> str:
        return self.key_of(path) if self.unit is None else row[self.unit]

    def keys(self) -> list[str]:
        """Every unit of the source, files read only where a column names the units."""
        if self.unit is None:
            return [self.key_of(path) for path in self.files()]
        return list(
            dict.fromkeys(
                self.unit_of(path, row) for path in self.files() for row in self.rows(path)
            )
        )

    def is_reading(self, row: dict[str, str]) -> bool:
        if row[self.channel] in self.skip:
            return False
        return self.value is None or row[self.value] not in self.missing

    def instant(self, stamp: str) -> float:
        if not self.clock:
            return float(stamp)
        hours, _, minutes = stamp.partition(":")
        return int(hours) + int(minutes) / _MINUTES_PER_HOUR

    def times_of(self, keys: set[str]) -> dict[str, Times]:
        """The instants of each channel of the units named, in one pass over the files."""
        collected: dict[str, dict[str, list[float]]] = {key: defaultdict(list) for key in keys}
        for path in self.files():
            if self.unit is None and self.key_of(path) not in keys:
                continue
            for row in self.rows(path):
                key = self.unit_of(path, row)
                if key in collected and self.is_reading(row):
                    collected[key][row[self.channel]].append(self.instant(row[self.time]))
        return {
            key: {channel: np.asarray(t, dtype=np.float64) for channel, t in channels.items()}
            for key, channels in collected.items()
        }


def audit(name: str, table: LongTable, units: int, seed: int) -> tuple[list[UnitRow], int]:
    """The rows of the units drawn that hold a reading, and how many drawn hold none."""
    drawn = sorted(chosen_keys(table.keys(), units, seed))
    times = table.times_of(set(drawn))
    read = [(key, times[key]) for key in drawn if times[key]]
    return unit_rows_of(name, read), len(drawn) - len(read)


TABLE_MEASURES = (
    ("co_observation", "Co-observed"),
    ("gap_variation", "Gap variation"),
    ("rate_spread", "Rate spread"),
    ("channel_share", "Channels / unit"),
)


def summary_table(name: str, rows: Sequence[UnitRow], empty: int) -> list[str]:
    """The units read, the median channels and readings per unit, and every unit measure.

    A measure is its median with its 10th and 90th percentiles in brackets.
    """
    head = ["Source", "Units", "Empty", "Channels", "Readings", *(t for _, t in TABLE_MEASURES)]
    cells = [name, str(len(rows)), str(empty)]
    for counted in ("channels", "observations"):
        cells.append(
            f"{statistics.median(getattr(row, counted) for row in rows):,.0f}" if rows else "—"
        )
    for measure, _ in TABLE_MEASURES:
        low, middle, high = quantiles(float(getattr(row, measure)) for row in rows)
        cells.append("—" if math.isnan(middle) else f"{middle:.2f} [{low:.2f}; {high:.2f}]")
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        "| " + " | ".join(cells) + " |",
    ]


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="the source, as the table names it")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", required=True, help="the files, as a glob under the root")
    units = parser.add_mutually_exclusive_group(required=True)
    units.add_argument("--unit", help="the column naming a reading's unit")
    units.add_argument("--unit-from-file", action="store_true", help="each file is a unit")
    parser.add_argument("--channel", required=True, help="the column naming a reading's channel")
    parser.add_argument("--time", required=True, help="the column holding a reading's instant")
    parser.add_argument("--clock", action="store_true", help="instants are HH:MM, read in hours")
    parser.add_argument("--value", help="the column holding a reading's value")
    parser.add_argument("--missing", nargs="*", default=[], help="values of no reading")
    parser.add_argument("--skip", nargs="*", default=[], help="channels that are no reading")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--units", type=int, default=100, help="units drawn")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/report/raw-regime-audit")
    arguments = parser.parse_args()
    if arguments.missing and arguments.value is None:
        parser.error("--missing needs --value")
    long_table = LongTable(
        root=arguments.root,
        pattern=arguments.pattern,
        channel=arguments.channel,
        time=arguments.time,
        unit=arguments.unit,
        clock=arguments.clock,
        value=arguments.value,
        missing=frozenset(arguments.missing),
        skip=frozenset(arguments.skip),
        delimiter=arguments.delimiter,
    )
    rows, empty = audit(arguments.name, long_table, arguments.units, arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / f"{arguments.name}.csv", UNIT_COLUMNS, [asdict(row) for row in rows])
    for line in summary_table(arguments.name, rows, empty):
        print(line)


if __name__ == "__main__":
    main()
