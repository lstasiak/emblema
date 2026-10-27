"""Measure how the Mars Express power telemetry is sampled, month by month, from its raw files.

The power lines are published as a wide table, one column per line at shared instants, and the
solar aspect angles as another at instants of their own; read as a long table the three training
Martian years would be some 190 million rows. Every calendar month (UTC) of those years is a unit
and all of them are measured, so this script hands each month's instants per channel straight to
the coverage report's unit measures (``regime_coverage_report.unit_rows_of``) instead of writing
the rows out for the raw audit, and prints the raw audit's table.

    uv run scripts/mex_power_audit.py --root data/raw/mex/mars-express-power-3years \
        --out data/report/raw-regime-audit

The test set's year is not read. A cell that is empty or not a finite number is no reading.
Writes ``mex.csv`` with the coverage report's unit columns.
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.raw_regime_audit import summary_table
from scripts.regime_coverage_report import UNIT_COLUMNS, Times, UnitRow, unit_rows_of, write

MILLISECONDS_PER_HOUR = 3_600_000.0
PATTERNS = ("train_set/power--*.csv", "train_set/context--*--saaf.csv")


def months_of(stamps: NDArray[np.int64]) -> NDArray[np.str_]:
    """The calendar month (UTC) of each instant in milliseconds since the epoch, ``YYYY-MM``."""
    return stamps.astype("datetime64[ms]").astype("datetime64[M]").astype(str)


def collect(paths: Iterable[Path]) -> dict[str, dict[str, list[NDArray[np.float64]]]]:
    """Each month's instants per channel, in hours since the epoch, over the files given."""
    import pandas as pd

    held: dict[str, dict[str, list[NDArray[np.float64]]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        frame = pd.read_csv(path)
        stamps = frame["ut_ms"].to_numpy(dtype=np.int64)
        months = months_of(stamps)
        hours = stamps / MILLISECONDS_PER_HOUR
        for channel in frame.columns.drop("ut_ms"):
            values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=np.float64)
            read = np.isfinite(values)
            for month in np.unique(months[read]):
                held[str(month)][channel].append(hours[read & (months == month)])
        print(f"{path.name}: {len(frame):,} rows", flush=True)
    return held


def measure(root: Path) -> list[UnitRow]:
    paths = sorted(path for pattern in PATTERNS for path in root.glob(pattern))
    held = collect(paths)
    read: list[tuple[str, Times]] = [
        (f"mex/{month}", {channel: np.concatenate(parts) for channel, parts in channels.items()})
        for month, channels in sorted(held.items())
    ]
    return unit_rows_of("mex", read)


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    rows = measure(arguments.root)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "mex.csv", UNIT_COLUMNS, [asdict(row) for row in rows])
    for line in summary_table("mex", rows, 0):
        print(line)


if __name__ == "__main__":
    main()
