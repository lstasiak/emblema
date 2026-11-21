r"""The turbines read as new systems: the published forecaster against the classical answers.

For one farm, the cells of every start and day are joined from the files that wrote them
(``fleet_cold_start_cells.py``, ``local_forecaster_answers.py --flat``,
``fleet_chronos2_answers.py``, ``fleet_tree_answers.py``) and read as the satellites' cold start
was read (``cold_start_reading.py``): each window's mean over its cells of the squared error of
the hour's mean, a window being one turbine's hour; for a comparison the geometric mean over
windows of one answer's window mean over another's; against a bar of several answers, the one
the numerator leads least, taken in every draw. The draws resample the starts, whole: a start's
turbines share their weather and its days follow one another, so the start is the unit, as a
farm and a date.

Two bars of classical answers: one without the farm, of what the new turbine alone gives — the
answers that learn nothing, the local forecasters and the ridges on its own hours since the
start, as the satellites' bar was; and one with the farm, which adds the ridges and trees fitted
on the other turbines before the start. The normal-behaviour bars add the answers that know the
forecast hour's operating conditions, and their comparisons read only the cells where those
conditions were read. The ridges on the turbine's own hours join a bar only where they answer
every cell of the comparison, since they cannot answer before they have rows enough; every other
answer of a bar always keeps its place, and the comparison reads the cells all of them answer.

- The verdicts read the first 30 days of every start, days 2, 7 and 14 together, each a level
  against one and the least gain: the published forecaster on 47 hours against the bar with the
  farm; with the other turbines' mean against it on 47 hours alone; and of normal behaviour,
  with the hour's operating conditions, against each normal-behaviour bar.
- The curve reads each day alone, on the starts whose year of days lies in the years read only,
  so every day of it is read on the same starts.
- Beside, descriptively: the forecaster on 47 hours against the bar without the farm, as the
  satellites were read; its other rungs; the farm's normal-behaviour trees against the bar
  without the farm, the room a model that knew the turbine's kind has; the forecaster of normal
  behaviour on every hour since the start, against both bars and on 47 hours, which the curve
  reads as the turbine's history grows; the ridges on the turbine's own hours against the
  farm's; the oracle that knows the hour's power; the variant with the pitch motors' currents;
  and the verdicts within each group of temperatures, where a
  window an answer hits exactly, as a quantised sensor's standing hour can be, is left out and
  counted.

Beside each ratio, the median over cells of the cell's squared error over the bar's, each ridge
on the turbine's own hours left out with the share it answers, and the share of the cells the
bar leaves unanswered.

    uv run scripts/fleet_cold_start_reading.py --cells DIR --farm kelmarsh \\
        --least-gain 0.8 [--resamples 10000] [--seed 1] --out DIR/read

Reads ``starts-<farm>.csv`` and ``answers-``, ``local-``, ``chronos2-`` and
``trees-<farm>-N<days>.csv`` from ``--cells``; writes ``comparisons.csv`` and ``losses.csv``
under ``--out`` and prints both.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cold_start_cells import NO_LEARNING  # noqa: E402
from scripts.cold_start_reading import EXACT, Day, geometric, joined, pooled  # noqa: E402
from scripts.fleet_chronos2_answers import RUNGS  # noqa: E402
from scripts.fleet_cold_start_cells import (  # noqa: E402
    CURVE_DAYS,
    FLEET,
    FLEET_NBM,
    FLEET_NBM_CURRENTS,
    LOCAL_NBM,
    ORACLE,
    OWN,
    OWN_NBM,
    VERDICT_DAYS,
)
from scripts.fleet_hours import TARGETS, parsed_axis  # noqa: E402
from scripts.fleet_tree_answers import TREES  # noqa: E402
from scripts.history_forecast_reading import interval, level_of, markdown, write  # noqa: E402
from scripts.local_forecaster_answers import FLAT, FORECASTERS  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Cells = NDArray[np.bool_]
PAST_ALONE = (*NO_LEARNING, *FORECASTERS, *FLAT, *OWN)
PAST_FLEET = (*PAST_ALONE, *FLEET, "fleet_trees")
NBM_ALONE = (*PAST_ALONE, LOCAL_NBM, OWN_NBM)
NBM_FLEET = (*PAST_FLEET, LOCAL_NBM, OWN_NBM, FLEET_NBM, "fleet_nbm_trees")
NBM_CURRENTS = (*NBM_FLEET, FLEET_NBM_CURRENTS, "fleet_nbm_trees_currents")
BARS = {
    "without the farm": PAST_ALONE,
    "with the farm": PAST_FLEET,
    "normal behaviour without the farm": NBM_ALONE,
    "normal behaviour with the farm": NBM_FLEET,
    "normal behaviour with the farm and currents": NBM_CURRENTS,
}
CLASSICAL = NBM_CURRENTS
# The answers fitted on the turbine's own hours, which join a bar only where they answer every
# cell of the comparison.
OWN_HISTORY = (*OWN, OWN_NBM)
NBM = "chronos2_nbm"
NBM_WITH_CURRENTS = "chronos2_nbm_currents"
NBM_OWN = "chronos2_nbm_own"


@dataclass(frozen=True)
class Comparison:
    """One answer against another, or against the best of a named bar.

    Attributes:
        of: The answer read.
        against: The answer it is read against, or the bar's name in ``BARS``.
        scope: The answer whose cells the comparison reads; every cell when empty.
        verdict: Whether the first month reads a level for it.
    """

    of: str
    against: str
    scope: str = ""
    verdict: bool = False

    @property
    def bar(self) -> tuple[str, ...]:
        return BARS.get(self.against, (self.against,))


VERDICTS = (
    Comparison("chronos2_47", "with the farm", verdict=True),
    Comparison("chronos2_neighbours", "chronos2_47", verdict=True),
    Comparison(NBM, "normal behaviour without the farm", NBM, verdict=True),
    Comparison(NBM, "normal behaviour with the farm", NBM, verdict=True),
)
COMPARISONS = (
    *VERDICTS,
    Comparison("chronos2_47", "without the farm"),
    Comparison("chronos2_47_each", "with the farm"),
    Comparison("chronos2_own", "with the farm"),
    Comparison("chronos2_own", "chronos2_47"),
    Comparison("chronos2_neighbours_concurrent", "chronos2_47"),
    Comparison("fleet_nbm_trees", "normal behaviour without the farm", NBM),
    Comparison(NBM, "chronos2_47", NBM),
    Comparison(NBM_WITH_CURRENTS, "normal behaviour with the farm and currents", NBM_WITH_CURRENTS),
    Comparison(NBM_WITH_CURRENTS, NBM, NBM_WITH_CURRENTS),
    Comparison(NBM_OWN, "normal behaviour without the farm", NBM_OWN),
    Comparison(NBM_OWN, "normal behaviour with the farm", NBM_OWN),
    Comparison(NBM_OWN, NBM, NBM_OWN),
    Comparison("ridge", "fleet_ridge"),
    Comparison("window_ridge", "fleet_window_ridge"),
    Comparison(OWN_NBM, FLEET_NBM, NBM),
    Comparison("fleet_trees", "fleet_window_ridge"),
    Comparison("fleet_nbm_trees", FLEET_NBM, NBM),
    Comparison(ORACLE, "fleet_window_ridge"),
    Comparison(ORACLE, "with the farm"),
)
GROUPED = (VERDICTS[0], VERDICTS[2], VERDICTS[3])
# The temperatures by what they measure, for the verdict's comparison within each.
GROUPS = {
    "stator": ("Stator temperature 1",),
    "main bearings": (
        "Front bearing temperature",
        "Rear bearing temperature",
        "Rotor bearing temp",
    ),
    "generator bearings": (
        "Generator bearing front temperature",
        "Generator bearing rear temperature",
    ),
    "pitch motors": (
        "Temperature motor axis 1",
        "Temperature motor axis 2",
        "Temperature motor axis 3",
    ),
    "gear oil": ("Gear oil inlet temperature", "Gear oil temperature"),
    "transformer": ("Transformer temperature", "Transformer cell temperature"),
    "cabinets and nacelle": (
        "CPU temperature",
        "Temp. top box",
        "Nacelle temperature",
        "Hub temperature",
    ),
}
COMPARISON_COLUMNS = (
    "farm",
    "set",
    "days",
    "of",
    "against",
    "starts",
    "windows",
    "cells",
    "ratio",
    "low",
    "high",
    "cell_median",
    "left_out",
    "level",
)
LOSS_COLUMNS = ("farm", "days", "answer", "windows", "cells", "geometric_mean", "low", "high")


@dataclass(frozen=True)
class Windows:
    """A day's cells grouped into windows, one turbine's hour each, and each window's start."""

    index: Ints
    starts: Ints

    @classmethod
    def of(cls, day: Day, cells: Cells) -> Windows:
        keys = [(k[1], k[0]) for k, kept in zip(day.keys, cells, strict=True) if kept]
        names = sorted(set(keys))
        position = {key: n for n, key in enumerate(names)}
        dates = sorted({parsed_axis(axis)[1] for axis, _ in names})
        starts = np.asarray(
            [dates.index(parsed_axis(axis)[1]) for axis, _ in names], dtype=np.int64
        )
        return cls(np.asarray([position[key] for key in keys], dtype=np.int64), starts)

    def losses(self, errors: Floats) -> Floats:
        sums = np.bincount(self.index, weights=errors, minlength=len(self.starts))
        counts = np.bincount(self.index, minlength=len(self.starts))
        out: Floats = sums / counts
        return out

    def weights(self, resamples: int, seed: int) -> Floats:
        """``[resamples, windows]``: how often each window's start is drawn, with replacement."""
        count = int(self.starts.max()) + 1
        drawn = np.random.default_rng(seed).integers(0, count, (resamples, count))
        counts = np.stack([np.bincount(row, minlength=count) for row in drawn])
        out: Floats = counts[:, self.starts].astype(np.float64)
        return out


def restricted(day: Day, keep: Cells) -> Day:
    """The day's cells ``keep`` marks."""
    return Day(
        day.mission,
        day.days,
        tuple(k for k, kept in zip(day.keys, keep, strict=True) if kept),
        day.target[keep],
        {name: values[keep] for name, values in day.answers.items()},
    )


def comparison_rows(
    day: Day,
    comparisons: Sequence[Comparison],
    *,
    reading: str,
    levels: bool,
    least_gain: float,
    resamples: int,
    seed: int,
    drop_exact: bool = False,
) -> list[dict[str, object]]:
    """The comparisons' rows on the day's cells; with ``levels``, the verdicts read a level.

    A window an answer hits exactly leaves no ratio, and is refused; with ``drop_exact``, as for
    a few channels read alone, whose hourly mean can stand still on a quantised sensor, it is
    left out of that comparison and counted.

    Raises:
        ValueError: If an answer hits a window exactly and ``drop_exact`` is not set.
    """
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        of, against = comparison.of, comparison.bar
        read = day
        if comparison.scope:
            if comparison.scope not in day.answers:
                continue
            read = restricted(day, day.covered([comparison.scope]))
        if of not in read.answers or not read.keys:
            continue
        named = comparison.against in BARS
        present = [a for a in against if a in read.answers]
        bar = (
            [a for a in present if a not in OWN_HISTORY or read.covered([a]).all()]
            if named
            else present
        )
        left_out = [f"{a} {read.covered([a]).mean():.3f}" for a in present if a not in bar]
        if not bar:
            continue
        cells = read.covered([of, *bar])
        if not cells.any():
            continue
        unanswered = 1.0 - cells.sum() / read.covered([of]).sum()
        if unanswered > 0:
            left_out.append(f"{unanswered:.3f} of the cells unanswered by the bar")
        windows = Windows.of(read, cells)
        error = (read.answers[of][cells] - read.target[cells]) ** 2
        errors = [(read.answers[a][cells] - read.target[cells]) ** 2 for a in bar]
        top, bottoms = windows.losses(error), [windows.losses(e) for e in errors]
        kept = np.all([top > 0, *(b > 0 for b in bottoms)], axis=0)
        if drop_exact and not kept.all():
            left_out.append(f"{int((~kept).sum())} windows hit exactly")
        else:
            kept[:] = True
        ratio, low, high, best = geometric(
            top[kept],
            [b[kept] for b in bottoms],
            windows.weights(resamples, seed)[:, kept],
        )
        cell_median = float(np.median((error + EXACT) / (errors[best] + EXACT)))
        rows.append(
            {
                "farm": day.mission,
                "set": reading,
                "days": day.days,
                "of": of,
                "against": (
                    f"{comparison.against}, best of {len(bar)}: {bar[best]}" if named else bar[0]
                ),
                "starts": int(windows.starts.max()) + 1,
                "windows": len(windows.starts),
                "cells": int(cells.sum()),
                "ratio": ratio,
                "low": low,
                "high": high,
                "cell_median": cell_median,
                "left_out": "; ".join(left_out),
                "level": (
                    level_of(low, high, least_gain)
                    if levels and comparison.verdict
                    else "descriptive"
                ),
            }
        )
    return rows


def loss_rows(day: Day, resamples: int, seed: int) -> list[dict[str, object]]:
    """Each answer's geometric mean over windows of its window losses, on the cells it answers."""
    rows: list[dict[str, object]] = []
    for name in (*CLASSICAL, *RUNGS, ORACLE):
        if name not in day.answers:
            continue
        cells = day.covered([name])
        if not cells.any():
            continue
        windows = Windows.of(day, cells)
        losses = windows.losses((day.answers[name][cells] - day.target[cells]) ** 2)
        if (losses <= 0).any():
            raise ValueError(f"{name} missed by nothing on a window")
        logs = np.log(losses)
        weights = windows.weights(resamples, seed)
        low, high = interval(np.exp((weights @ logs) / weights.sum(axis=1)))
        rows.append(
            {
                "farm": day.mission,
                "days": day.days,
                "answer": name,
                "windows": len(windows.starts),
                "cells": int(cells.sum()),
                "geometric_mean": float(np.exp(logs.mean())),
                "low": low,
                "high": high,
            }
        )
    return rows


def channel_cells(day: Day, names: Sequence[str]) -> Cells:
    wanted = {TARGETS.index(n) for n in names}
    return np.asarray([k[2] in wanted for k in day.keys], dtype=np.bool_)


def curve_starts(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["start"] for row in csv.DictReader(handle) if row["curve"] == "1"}


def on_curve(day: Day, starts: set[str]) -> Cells:
    return np.asarray(
        [parsed_axis(k[1])[1].isoformat() in starts for k in day.keys], dtype=np.bool_
    )


def read(
    cells: Path, farm: str, *, least_gain: float, resamples: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The comparisons and losses of one farm's files.

    Raises:
        ValueError: If a day's file is missing, a day lacks an answer of the bar's or a rung
            of the forecaster's, or the first month lacks a verdict's answer.
    """
    days: dict[int, Day] = {}
    for number in CURVE_DAYS:
        label = f"{farm}-N{number}"
        files = [
            cells / f"{kind}-{label}.csv" for kind in ("answers", "local", "chronos2", "trees")
        ]
        if not files[0].exists() and number not in VERDICT_DAYS:
            continue
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise ValueError(f"{label}: {missing} not written")
        days[number] = joined(farm, str(number), files)
        absent = [n for n in (*FORECASTERS, *FLAT, *RUNGS, *TREES) if n not in days[number].answers]
        if absent:
            raise ValueError(f"{label}: no answer of {absent}")

    def compared(
        day: Day,
        comparisons: Sequence[Comparison],
        reading: str,
        *,
        levels: bool = False,
        drop_exact: bool = False,
    ) -> list[dict[str, object]]:
        return comparison_rows(
            day,
            comparisons,
            reading=reading,
            levels=levels,
            least_gain=least_gain,
            resamples=resamples,
            seed=seed,
            drop_exact=drop_exact,
        )

    month = pooled([days[n] for n in VERDICT_DAYS])
    comparisons = compared(month, COMPARISONS, "first month", levels=True)
    for group, names in GROUPS.items():
        within = restricted(month, channel_cells(month, names))
        comparisons += compared(within, GROUPED, f"group: {group}", drop_exact=True)
    starts = curve_starts(cells / f"starts-{farm}.csv")
    for day in (days[n] for n in sorted(days)):
        kept = on_curve(day, starts)
        if kept.any():
            comparisons += compared(restricted(day, kept), COMPARISONS, "curve")
    losses = loss_rows(month, resamples, seed)
    return comparisons, losses


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--farm", required=True)
    parser.add_argument("--least-gain", type=float, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    comparisons, losses = read(
        arguments.cells,
        arguments.farm,
        least_gain=arguments.least_gain,
        resamples=arguments.resamples,
        seed=arguments.seed,
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "comparisons.csv", COMPARISON_COLUMNS, comparisons)
    write(arguments.out / "losses.csv", LOSS_COLUMNS, losses)
    print(markdown(COMPARISON_COLUMNS, comparisons))
    print()
    print(markdown(LOSS_COLUMNS, losses))


if __name__ == "__main__":  # pragma: no cover
    main()
