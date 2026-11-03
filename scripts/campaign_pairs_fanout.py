"""Read many pairs of an export at once, each in a process of its own.

``campaign_pairs_report.py`` holds every answer it reads in memory, and a large export does not fit.
This script gives each pair or interaction its own directory, holding the export's answers filtered
to its sides, runs the report there, and merges the tables. The bootstrap draws afresh from the one
seed for every pair, so the merged tables are byte for byte those of a single run over all pairs,
at the report's default resamples, seed and floors.

A plan has one line per reading, its seven fields separated by tabs, since a candidate's name holds
commas:

    pair  control-campaigns  control  budget  candidate-campaigns  candidate  budget
    interaction  campaigns  taught  taught-control  candidate  control  budget

Campaigns of a side are joined by ``+``. Blank lines and lines starting with ``#`` are skipped.

    uv run scripts/campaign_pairs_fanout.py --plan plan.tsv --export DIR --out DIR [--jobs 8]
"""

import argparse
import csv
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPORT = Path(__file__).with_name("campaign_pairs_report.py")
CELLS, PREDICTIONS = "cells.csv", "predictions.csv"
MERGED = ("comparisons.csv", "interactions.csv")
FIELDS = 7


@dataclass(frozen=True, kw_only=True)
class Reading:
    """One pair or interaction of a plan.

    Attributes:
        arguments: The report's arguments for it, flag first.
        campaigns: The campaigns its sides read.
        candidates: The candidates its sides read.
    """

    arguments: tuple[str, ...]
    campaigns: frozenset[str]
    candidates: frozenset[str]


def read_plan(text: str) -> list[Reading]:
    """The readings of a plan, in its order.

    Raises:
        SystemExit: If a line is neither a pair nor an interaction of seven fields.
    """
    readings: list[Reading] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != FIELDS or fields[0] not in {"pair", "interaction"}:
            raise SystemExit(f"plan line {number} is not a reading of {FIELDS} fields: {line}")
        if fields[0] == "pair":
            campaigns = {*fields[1].split("+"), *fields[4].split("+")}
            candidates = {fields[2], fields[5]}
        else:
            campaigns = set(fields[1].split("+"))
            candidates = set(fields[2:6])
        readings.append(
            Reading(
                arguments=(f"--{fields[0]}", *fields[1:]),
                campaigns=frozenset(campaigns),
                candidates=frozenset(candidates),
            )
        )
    return readings


def split_answers(source: Path, readings: Sequence[Reading], directories: Sequence[Path]) -> None:
    """Write each reading's answers to its directory, in one pass over the export."""
    handles = [(directory / PREDICTIONS).open("w", newline="") for directory in directories]
    try:
        writers = [csv.writer(handle) for handle in handles]
        with source.open(newline="") as answers:
            rows = csv.reader(answers)
            header = next(rows)
            campaign_at, candidate_at = header.index("campaign"), header.index("candidate")
            for writer in writers:
                writer.writerow(header)
            for row in rows:
                for reading, writer in zip(readings, writers, strict=True):
                    if row[campaign_at] in reading.campaigns and (
                        row[candidate_at] in reading.candidates
                    ):
                        writer.writerow(row)
    finally:
        for handle in handles:
            handle.close()


def run_report(directory: Path, reading: Reading) -> int:
    """Run the report over one directory, its output in ``pairs.log``: the exit status."""
    with (directory / "pairs.log").open("w") as log:
        return subprocess.run(
            [sys.executable, str(REPORT), "--out", str(directory), *reading.arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def merge(name: str, directories: Iterable[Path], target: Path) -> int:
    """The header once, then every directory's rows in order: how many rows, none for no table."""
    lines: list[bytes] = []
    for directory in directories:
        table = directory / name
        if table.exists():
            # In bytes, as a CSV's line ends are its writer's and must survive the merge.
            read = table.read_bytes().splitlines(keepends=True)
            lines += read if not lines else read[1:]
    if not lines:
        return 0
    target.write_bytes(b"".join(lines))
    return len(lines) - 1


def main(argv: Sequence[str] | None = None) -> None:
    """Split the export, run the readings side by side and merge what they wrote.

    Raises:
        SystemExit: If a reading failed; the merged tables are then not written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--jobs", type=int, default=8)
    arguments = parser.parse_args(argv)
    readings = read_plan(arguments.plan.read_text())
    if not readings:
        raise SystemExit(f"no reading in {arguments.plan}")
    for name in MERGED:
        (arguments.out / name).unlink(missing_ok=True)
    jobs: Path = arguments.out / "jobs"
    shutil.rmtree(jobs, ignore_errors=True)
    directories = [jobs / f"{index:03d}" for index in range(len(readings))]
    for directory in directories:
        directory.mkdir(parents=True)
        shutil.copyfile(arguments.export / CELLS, directory / CELLS)
    split_answers(arguments.export / PREDICTIONS, readings, directories)
    with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        statuses = list(pool.map(run_report, directories, readings))
    failed = [str(d) for d, status in zip(directories, statuses, strict=True) if status != 0]
    if failed:
        raise SystemExit(f"readings failed, see pairs.log in: {' '.join(failed)}")
    for name in MERGED:
        rows = merge(name, directories, arguments.out / name)
        print(f"{name}: {rows} rows")


if __name__ == "__main__":  # pragma: no cover
    main()
