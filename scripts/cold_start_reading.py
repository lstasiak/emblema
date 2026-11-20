r"""The cold-start forecasts read: the published forecaster against the classical answers, by rung.

For each mission read as a new system and each day N, the cells' answers are joined from the
files that wrote them (``cold_start_cells.py``, ``local_forecaster_answers.py``,
``chronos2_ladder.py``) and read as the published forecaster was read on 2026-11-19: each
window's mean over its cells of the squared error of the hour's mean, and for a comparison the
geometric mean over windows of one answer's window mean over another's; against a bar of several
answers, the one the numerator leads least, taken in every draw, so the bar is never weaker than
any of its answers. The draws resample whole days, since a day's hours share their context. An
answer joins the classical bar on a day only if it answers every cell of it: the ridges do not
answer before they have rows enough.

The comparisons, each on the cells both sides answer:

- ``chronos2_47``, ``chronos2_47_each`` and ``chronos2_own`` against the best classical answer,
  and the latter two against the first;
- the rungs on the cells with a counterpart: each library series against the 47 hours alone,
  and each rung against the one below it (the same kind against another kind, the nearest series
  against a random one of the same kind); beside them the oracle, the target's own next hour
  handed as a library series is, the most that path could give.

Beside each ratio, the median over cells of the cell's squared error over the bar's: a window's
mean is carried by its worst cell, the median by the typical one.

Each day is read on its own, descriptively. The days named by ``--pool``, whose hours are
disjoint, are also read together, and on them alone the comparisons named by ``--verdict`` read
a level, against one and the least gain; there an answer joins the classical bar only if it
answers every cell of every pooled day.

    uv run scripts/cold_start_reading.py --cells DIR --mission ESA-Mission2 \\
        --days 2 7 14 30 60 90 180 365 --pool 2 7 14 \\
        --verdict chronos2_47 classical --verdict chronos2_analog chronos2_47 \\
        --least-gain 0.8 [--resamples 10000] --out DIR/read

Reads ``answers-``, ``local-`` and ``chronos2-<mission>-N<days>.csv`` from ``--cells``; writes
``comparisons.csv`` and ``losses.csv`` under ``--out`` and prints both.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cold_start_cells import LEARNT, NO_LEARNING  # noqa: E402
from scripts.history_forecast_reading import (  # noqa: E402
    interval,
    level_of,
    markdown,
    write,
)
from scripts.local_forecaster_answers import FORECASTERS  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Key = tuple[int, str, int]
HOURS_PER_DAY = 24
# A cell's squared error is floored at this in a ratio of two, so an answer exact to the digit
# leaves a ratio; far below any error the answers make in normalised units.
EXACT = 1e-12
CLASSICAL = (*NO_LEARNING, *FORECASTERS, *LEARNT)
PUBLISHED = ("chronos2_47", "chronos2_47_each", "chronos2_own")
LADDER = ("chronos2_unrelated", "chronos2_kind", "chronos2_analog", "chronos2_oracle")
COMPARISONS = (
    ("chronos2_47", CLASSICAL),
    ("chronos2_47_each", CLASSICAL),
    ("chronos2_own", CLASSICAL),
    ("chronos2_47_each", ("chronos2_47",)),
    ("chronos2_own", ("chronos2_47",)),
    ("chronos2_unrelated", ("chronos2_47",)),
    ("chronos2_kind", ("chronos2_47",)),
    ("chronos2_analog", ("chronos2_47",)),
    ("chronos2_kind", ("chronos2_unrelated",)),
    ("chronos2_analog", ("chronos2_kind",)),
    ("chronos2_oracle", ("chronos2_47",)),
)
COMPARISON_COLUMNS = (
    "mission",
    "days",
    "of",
    "against",
    "windows",
    "cells",
    "ratio",
    "low",
    "high",
    "cell_median",
    "level",
)
LOSS_COLUMNS = ("mission", "days", "answer", "windows", "cells", "geometric_mean", "low", "high")


@dataclass(frozen=True)
class Day:
    """One mission's cells on a day or several pooled, named by ``days``.

    The target and every answer's forecast per cell, NaN where an answer gave none.
    """

    mission: str
    days: str
    keys: tuple[Key, ...]
    target: Floats
    answers: Mapping[str, Floats]

    def covered(self, names: Sequence[str]) -> NDArray[np.bool_]:
        """The cells every one of ``names`` answers."""
        return np.all([np.isfinite(self.answers[n]) for n in names], axis=0)

    def window_losses(self, name: str, cells: NDArray[np.bool_]) -> tuple[Floats, Ints]:
        """Each window's mean squared error of ``name`` over ``cells``, and the windows' hours."""
        windows = np.asarray([k[0] for k in self.keys], dtype=np.int64)[cells]
        errors = (self.answers[name][cells] - self.target[cells]) ** 2
        order, inverse = np.unique(windows, return_inverse=True)
        sums = np.bincount(inverse, weights=errors)
        return sums / np.bincount(inverse), order


def read_rows(path: Path) -> dict[Key, dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (int(row["window"]), row["axis"], int(row["channel"])): row
            for row in csv.DictReader(handle)
        }


def joined(mission: str, days: str, files: Sequence[Path]) -> Day:
    """The cells of the first file with every ``mean_<name>`` column of all of them.

    Raises:
        ValueError: If a file holds other cells than the first, or a name twice.
    """
    first = read_rows(files[0])
    keys = tuple(first)
    answers: dict[str, Floats] = {}
    for path in files:
        rows = first if path == files[0] else read_rows(path)
        if set(rows) != set(keys):
            raise ValueError(f"{path.name} holds other cells than {files[0].name}")
        names = [
            c.removeprefix("mean_") for c in next(iter(rows.values())) if c.startswith("mean_")
        ]
        for name in names:
            if name in answers:
                raise ValueError(f"{name} is answered by two files")
            answers[name] = np.asarray(
                [float(rows[k][f"mean_{name}"] or "nan") for k in keys], dtype=np.float64
            )
    target = np.asarray([float(first[k]["target_mean"]) for k in keys], dtype=np.float64)
    return Day(mission, days, keys, target, answers)


def day_weights(windows: Ints, resamples: int, seed: int) -> Floats:
    """``[resamples, windows]``: how often each window's day is drawn, whole days with replacement.

    The hours of one day share most of their context, so their errors are not independent; a
    day is the unit resampled, as a block bootstrap resamples blocks.
    """
    days, inverse = np.unique(windows // HOURS_PER_DAY, return_inverse=True)
    drawn = np.random.default_rng(seed).integers(0, len(days), (resamples, len(days)))
    counts = np.stack([np.bincount(row, minlength=len(days)) for row in drawn])
    weights: Floats = counts[:, inverse].astype(np.float64)
    return weights


def geometric(
    numerator: Floats, bars: Sequence[Floats], weights: Floats
) -> tuple[float, float, float, int]:
    """The geometric mean over windows of ``numerator`` over the bar it leads least.

    Returns the ratio, its 95 % interval over the draws ``weights`` makes (``day_weights``),
    the least-led bar taken in every draw, and which bar that is on all windows.

    Raises:
        ValueError: If a window's loss is nought on either side, which leaves no ratio.
    """
    if (numerator <= 0).any() or any((bar <= 0).any() for bar in bars):
        raise ValueError("an answer missed by nothing on a window: no ratio")
    logs = np.stack([np.log(numerator / bar) for bar in bars])
    points = logs.mean(axis=1)
    best = int(np.argmax(points))
    boots = ((weights @ logs.T) / weights.sum(axis=1, keepdims=True)).max(axis=1)
    low, high = interval(np.exp(boots))
    return float(np.exp(points[best])), low, high, best


def pooled(days: Sequence[Day]) -> Day:
    """The cells of several days as one, each answer NaN on a day that lacks it.

    Raises:
        ValueError: If two days hold the same cell, which their disjoint hours rule out.
    """
    keys = tuple(k for day in days for k in day.keys)
    if len(set(keys)) != len(keys):
        raise ValueError("two pooled days hold the same cell")
    names = sorted({name for day in days for name in day.answers})
    answers = {
        name: np.concatenate(
            [day.answers.get(name, np.full(len(day.keys), np.nan)) for day in days]
        )
        for name in names
    }
    label = f"{days[0].days}-{days[-1].days}"
    target = np.concatenate([day.target for day in days])
    return Day(days[0].mission, label, keys, target, answers)


def comparison_rows(
    day: Day,
    *,
    verdicts: set[tuple[str, str]],
    least_gain: float,
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for of, against in COMPARISONS:
        if of not in day.answers:
            continue
        everywhere = [a for a in against if a in day.answers and day.covered([a]).all()]
        bar = everywhere if against == CLASSICAL else [a for a in against if a in day.answers]
        if not bar:
            continue
        cells = day.covered([of, *bar])
        if not cells.any():
            continue
        top, windows = day.window_losses(of, cells)
        bottoms = [day.window_losses(a, cells)[0] for a in bar]
        weights = day_weights(windows, resamples, seed)
        ratio, low, high, best = geometric(top, bottoms, weights)
        error = (day.answers[of][cells] - day.target[cells]) ** 2
        against_error = (day.answers[bar[best]][cells] - day.target[cells]) ** 2
        cell_median = float(np.median((error + EXACT) / (against_error + EXACT)))
        named = (of, against[0] if len(against) == 1 else "classical")
        reads = named in verdicts
        rows.append(
            {
                "mission": day.mission,
                "days": day.days,
                "of": of,
                "against": bar[0] if len(bar) == 1 else f"best of {len(bar)}: {bar[best]}",
                "windows": len(windows),
                "cells": int(cells.sum()),
                "ratio": ratio,
                "low": low,
                "high": high,
                "cell_median": cell_median,
                "level": level_of(low, high, least_gain) if reads else "descriptive",
            }
        )
    return rows


def loss_rows(day: Day, resamples: int, seed: int) -> list[dict[str, object]]:
    """Each answer's geometric mean over windows of its window losses, on the cells it answers."""
    rows: list[dict[str, object]] = []
    for name in (*CLASSICAL, *PUBLISHED, *LADDER):
        if name not in day.answers:
            continue
        cells = day.covered([name])
        if not cells.any():
            continue
        losses, windows = day.window_losses(name, cells)
        if (losses <= 0).any():
            raise ValueError(f"{name} missed by nothing on a window")
        logs = np.log(losses)
        weights = day_weights(windows, resamples, seed)
        low, high = interval(np.exp((weights @ logs) / weights.sum(axis=1)))
        rows.append(
            {
                "mission": day.mission,
                "days": day.days,
                "answer": name,
                "windows": len(windows),
                "cells": int(cells.sum()),
                "geometric_mean": float(np.exp(logs.mean())),
                "low": low,
                "high": high,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--days", type=int, nargs="+", required=True)
    parser.add_argument(
        "--verdict", nargs=2, action="append", default=[], metavar=("OF", "AGAINST")
    )
    parser.add_argument("--pool", type=int, nargs="*", default=[], help="days read together")
    parser.add_argument("--least-gain", type=float, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if not set(arguments.pool) <= set(arguments.days):
        raise SystemExit(f"the pooled days {arguments.pool} are not all read")
    named = {(of, against[0] if len(against) == 1 else "classical") for of, against in COMPARISONS}
    unknown = [tuple(v) for v in arguments.verdict if tuple(v) not in named]
    if unknown or (arguments.verdict and not arguments.pool):
        raise SystemExit(f"verdicts {unknown or arguments.verdict} name no comparison of a pool")
    comparisons: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    read: dict[int, Day] = {}
    for days in arguments.days:
        label = f"{arguments.mission}-N{days}"
        files = [
            arguments.cells / f"{kind}-{label}.csv" for kind in ("answers", "local", "chronos2")
        ]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{label}: {missing} not written")
        read[days] = joined(arguments.mission, str(days), files)
    days_read = [read[days] for days in arguments.days]
    if arguments.pool:
        days_read.append(pooled([read[days] for days in arguments.pool]))
    for day in days_read:
        is_pool = arguments.pool and day is days_read[-1]
        comparisons += comparison_rows(
            day,
            verdicts={tuple(v) for v in arguments.verdict} if is_pool else set(),
            least_gain=arguments.least_gain,
            resamples=arguments.resamples,
            seed=arguments.seed,
        )
        losses += loss_rows(day, arguments.resamples, arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "comparisons.csv", COMPARISON_COLUMNS, comparisons)
    write(arguments.out / "losses.csv", LOSS_COLUMNS, losses)
    print(markdown(COMPARISON_COLUMNS, comparisons))
    print()
    print(markdown(LOSS_COLUMNS, losses))


if __name__ == "__main__":  # pragma: no cover
    main()
