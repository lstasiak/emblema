r"""The forecaster's answers corrected by a new turbine's own hours, as the farm's trees are.

The farm's trees are read beside their answers corrected by the new turbine's own residuals since
its start (``fleet_history_answers.py``); here the forecaster is handed the same corrections, so
that the two sides read the turbine's hours alike. ``hours`` writes, per farm, every own hour of
each new turbine before its last cell, as cells the forecaster answers as it answers any
(``fleet_chronos2_adapted.py answer --cells own-<farm>.csv``): an hour whose channel would be
scored and whose operating conditions are read. ``correct`` reads those answers beside the
cells' and writes each cell's answer on each rung corrected each way by the residuals of the own
hours before it (``_offset``, ``_ridge``, ``_boosted``); the small trees read the forecaster's
answer and the hour's conditions.

    uv run scripts/fleet_forecaster_history.py hours --farm F --hourly DIR --cells DIR
    uv run scripts/fleet_forecaster_history.py correct --farm F --hourly DIR --cells DIR \\
        --answers DIR --arm ARM

``hours`` writes ``own-<farm>.csv`` into ``--cells``; ``correct`` reads ``<arm>-<farm>-N<days>.csv``
and ``<arm>-own-<farm>.csv`` from ``--answers`` and writes ``<arm>-history-<farm>-N<days>.csv``
there.
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_ladder import read_cells, write  # noqa: E402
from scripts.cold_start_cells import CONTEXT  # noqa: E402
from scripts.fleet_cold_start_cells import scorable  # noqa: E402
from scripts.fleet_history_answers import SUFFIXES, Fit, History, corrected  # noqa: E402
from scripts.fleet_hours import (  # noqa: E402
    CONDITIONS,
    TARGETS,
    Turbine,
    hour_of,
    loaded_farm,
    parsed_axis,
)

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Key = tuple[int, str, int]
RUNGS = ("_nbm", "_nbm_own")


def new_turbine(turbines: Mapping[str, Turbine], axis: str) -> tuple[int, Floats]:
    """The start an axis names and its turbine's hours, every one before the start unknown.

    Raises:
        ValueError: If the axis names a turbine not loaded.
    """
    name, day = parsed_axis(axis)
    if name not in turbines:
        raise ValueError(f"{axis} names a turbine not loaded")
    start = hour_of(day)
    return start, turbines[name].before(start).values


def own_keys(turbines: Mapping[str, Turbine], cells: Sequence[Key]) -> list[Key]:
    """Every own hour of each cell's new turbine, from its start to before its last cell.

    An hour is kept for a channel where a cell of it would be scored and the hour's operating
    conditions are read, the hours the farm's trees are corrected by.
    """
    ends: dict[str, int] = {}
    for hour, axis, _ in cells:
        ends[axis] = max(ends.get(axis, hour), hour)
    out: list[Key] = []
    for axis in sorted(ends):
        start, values = new_turbine(turbines, axis)
        first = max(start - CONTEXT, 0)
        scored = scorable(values[first : ends[axis]])[start - first :, : len(TARGETS)]
        known = np.isfinite(values[start : ends[axis]][:, list(CONDITIONS)]).all(axis=1)
        for offset, channel in zip(*np.nonzero(scored & known[:, None]), strict=True):
            out.append((start + int(offset), axis, int(channel)))
    return sorted(out, key=lambda k: (k[1], k[0], k[2]))


def history_of(values: Floats, answers: Mapping[int, float], channel: int, end: int) -> History:
    """The forecaster's own hours of one channel before ``end`` and what it missed in each."""
    kept = sorted(h for h in answers if h < end)
    hours = np.asarray(kept, dtype=np.int64)
    given = np.asarray([answers[h] for h in kept], dtype=np.float64)
    conditions = values[hours][:, list(CONDITIONS)]
    rows = np.column_stack([given, conditions])
    return History(hours, rows, given, conditions, values[hours, channel] - given)


def corrected_cells(
    turbines: Mapping[str, Turbine],
    cells: Mapping[Key, float],
    own: Mapping[Key, float],
    *,
    small: Fit,
    seed: int,
) -> dict[str, dict[Key, float]]:
    """Each cell's answer corrected each way by the own hours before it, by suffix."""
    out: dict[str, dict[Key, float]] = {suffix: {} for suffix in SUFFIXES}
    by_group: dict[tuple[str, int], list[Key]] = {}
    for key in cells:
        by_group.setdefault((key[1], key[2]), []).append(key)
    own_by_group: dict[tuple[str, int], dict[int, float]] = {}
    for (hour, axis, channel), value in own.items():
        own_by_group.setdefault((axis, channel), {})[hour] = value
    news: dict[str, tuple[int, Floats]] = {}
    for (axis, channel), keys in sorted(by_group.items()):
        if axis not in news:
            news = {axis: new_turbine(turbines, axis)}
        start, values = news[axis]
        mine = sorted(keys)
        hours = np.asarray([k[0] for k in mine], dtype=np.int64)
        given = np.asarray([cells[k] for k in mine], dtype=np.float64)
        history = history_of(
            values, own_by_group.get((axis, channel), {}), channel, int(hours.max())
        )
        conditions = values[hours][:, list(CONDITIONS)]
        rows = np.column_stack([given, conditions])
        made = corrected(
            history, hours, rows, given, conditions, start=start, small=small, seed=seed
        )
        for suffix, values_of in made.items():
            out[suffix].update(zip(mine, (float(v) for v in values_of), strict=True))
    return out


def read_answers(path: Path, column: str) -> dict[Key, float]:
    """One answer's value per cell, cells it left blank left out."""
    with path.open(newline="") as handle:
        return {
            (int(row["window"]), row["axis"], int(row["channel"])): float(row[column])
            for row in csv.DictReader(handle)
            if row[column]
        }


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("hours", "correct"))
    parser.add_argument("--farm", required=True)
    parser.add_argument("--hourly", type=Path, required=True, help="telemetry_hourly_means's")
    parser.add_argument("--cells", type=Path, required=True, help="fleet_cold_start_cells's")
    parser.add_argument("--answers", type=Path, help="the forecaster's answers")
    parser.add_argument("--arm", help="the forecaster's arm, as its answers name it")
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    turbines = {t.name: t for t in loaded_farm(arguments.hourly)}
    files = sorted(arguments.cells.glob(f"answers-{arguments.farm}-N*.csv"))
    if arguments.step == "hours":
        keys = own_keys(turbines, [k for path in files for k in read_cells(path)])
        write(arguments.cells / f"own-{arguments.farm}.csv", keys, {}, ())
        print(f"{len(keys)} own hours of {len({k[1] for k in keys})} new turbines")
        return
    if arguments.answers is None or arguments.arm is None:
        raise SystemExit("correct reads the forecaster's answers: --answers and --arm")
    from scripts.fleet_tree_answers import small_fit

    own_file = arguments.answers / f"{arguments.arm}-own-{arguments.farm}.csv"
    for path in files:
        label = path.stem.removeprefix("answers-")
        given = arguments.answers / f"{arguments.arm}-{label}.csv"
        out: dict[str, dict[Key, float]] = {}
        for rung in RUNGS:
            column = f"mean_{arguments.arm}{rung}"
            made = corrected_cells(
                turbines,
                read_answers(given, column),
                read_answers(own_file, column),
                small=small_fit(),
                seed=arguments.seed,
            )
            out |= {f"{arguments.arm}{rung}{suffix}": v for suffix, v in made.items()}
        keys = read_cells(path)
        write(arguments.answers / f"{arguments.arm}-history-{label}.csv", keys, out, tuple(out))
        print(f"{label}: {len(keys)} cells")


if __name__ == "__main__":  # pragma: no cover
    main()
