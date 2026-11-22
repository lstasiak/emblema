r"""The farm's boosted trees' answers for the cells of turbines read as new systems.

For each turbine read as new and each of its temperatures, trees are fitted on the farm's other
turbines' hours before the first of January of the start's year, and nothing of the new one, on
the rows the farm's ridges read (``fleet_cold_start_cells.py``):

- ``fleet_trees``: the channel's 47 hours and the window's other channels' latest hour, as the
  farm's window ridge;
- ``fleet_nbm_trees``: the channel's 47 hours, the power's last day and the six signals of the
  forecast hour's operating conditions, as the farm's normal-behaviour ridge;
- ``fleet_nbm_trees_currents``, beside: the same with the pitch motors' currents of the hour.

Trees as the project's classical bar takes them: 300 rounds at a learning rate of 0.1, histogram
splits, seeded. A fit reads at most 100,000 rows, drawn by seed where there are more, so that a
farm's trees cost an hour or so. A cell is answered only where every reading its rows need is
there. The trees run in a process without torch, whose OpenMP runtime clashes with XGBoost's on
macOS:

    uv run scripts/fleet_tree_answers.py --farm kelmarsh --hourly DIR --cells DIR [--seed 1]

Reads ``answers-<farm>-N<days>.csv`` from ``--cells`` and writes ``trees-<farm>-N<days>.csv``
beside them: window, axis, channel and each answer.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_ladder import read_cells, write  # noqa: E402
from scripts.cold_start_cells import ROWS_PER_FEATURE  # noqa: E402
from scripts.fleet_cold_start_cells import (  # noqa: E402
    FLEET,
    FLEET_NBM,
    FLEET_NBM_CURRENTS,
    FLEET_ROWS,
    Rows,
    latest,
    scorable,
)
from scripts.fleet_hours import TARGETS, Turbine, hour_of, loaded_farm, parsed_axis  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Mask = NDArray[np.bool_]
Key = tuple[int, str, int]
# Each tree answer and the farm's ridge whose rows it reads.
TREES = {
    "fleet_trees": FLEET[1],
    "fleet_nbm_trees": FLEET_NBM,
    "fleet_nbm_trees_currents": FLEET_NBM_CURRENTS,
}
ROWS: Mapping[str, Rows] = {
    tree: next(r for r in FLEET_ROWS if r.name == ridge) for tree, ridge in TREES.items()
}
MOST_ROWS = 100_000
# A model fitted on rows and targets, answering rows.
Fit = Callable[[Floats, Floats, int], Callable[[Floats], Floats]]


def year_start(day: date) -> int:
    return hour_of(date(day.year, 1, 1))


def fleet_rows(
    turbines: Sequence[Turbine],
    read: Mapping[str, tuple[Mask, Floats]],
    kind: Rows,
    channel: int,
    until: int,
) -> dict[str, tuple[Floats, Floats]]:
    """Per turbine, its rows of the channel before ``until`` and their targets.

    An hour is kept where the channel is scored and every reading the rows need is there.
    ``read`` holds each turbine's ``scorable`` and ``latest``, which read no hour after their
    own, so one pass serves every year.
    """
    out = {}
    for turbine in turbines:
        scored, recent = read[turbine.name]
        hours = np.flatnonzero(scored[:until, channel])
        rows = kind.of(turbine.values, recent, hours, channel)
        kept = np.isfinite(rows).all(axis=1)
        out[turbine.name] = (rows[kept], turbine.values[hours[kept], channel])
    return out


def drawn(rows: Floats, targets: Floats, seed: int) -> tuple[Floats, Floats]:
    """At most ``MOST_ROWS`` of the rows, drawn by ``seed``, in their order."""
    if len(rows) <= MOST_ROWS:
        return rows, targets
    keep = np.sort(np.random.default_rng(seed).choice(len(rows), MOST_ROWS, replace=False))
    return rows[keep], targets[keep]


def answered(
    turbines: Sequence[Turbine],
    cells: Sequence[Key],
    fit: Fit,
    *,
    seed: int,
    say: Callable[[str], None] = lambda _: None,
) -> dict[str, dict[Key, float]]:
    """Every tree answer for every cell it can answer.

    Raises:
        ValueError: If a cell names a turbine not loaded.
    """
    by_name = {t.name: t for t in turbines}
    groups: dict[tuple[str, int], list[Key]] = {}
    for key in cells:
        name, day = parsed_axis(key[1])
        if name not in by_name:
            raise ValueError(f"{key[1]} names a turbine not loaded")
        groups.setdefault((name, year_start(day)), []).append(key)
    out: dict[str, dict[Key, float]] = {tree: {} for tree in TREES}
    read = {t.name: (scorable(t.values), latest(t.values)) for t in turbines}
    started = time.monotonic()
    for until in sorted({u for _, u in groups}):
        for tree, kind in ROWS.items():
            for channel in range(len(TARGETS)):
                rows_of = fleet_rows(turbines, read, kind, channel, until)
                for (name, year), keys in groups.items():
                    if year != until:
                        continue
                    wanted = [k for k in keys if k[2] == channel]
                    others = [v for t, v in rows_of.items() if t != name]
                    if not wanted or not others:
                        continue
                    rows = np.concatenate([r for r, _ in others])
                    if len(rows) < ROWS_PER_FEATURE * rows.shape[1]:
                        continue
                    targets = np.concatenate([y for _, y in others])
                    model = fit(*drawn(rows, targets, seed + channel), seed)
                    answer_cells(out[tree], by_name[name], kind, channel, wanted, model)
            say(
                f"{tree} before {until}: {len(out[tree])} cells, {time.monotonic() - started:.0f} s"
            )
    return out


def answer_cells(
    out: dict[Key, float],
    turbine: Turbine,
    kind: Rows,
    channel: int,
    keys: Sequence[Key],
    model: Callable[[Floats], Floats],
) -> None:
    """The model's answer for each of the new turbine's cells whose rows are whole."""
    for axis in sorted({k[1] for k in keys}):
        start = hour_of(parsed_axis(axis)[1])
        new = turbine.before(start)
        mine = [k for k in keys if k[1] == axis]
        hours = np.asarray([k[0] for k in mine], dtype=np.int64)
        rows = kind.of(new.values, latest(new.values), hours, channel)
        kept = np.isfinite(rows).all(axis=1)
        if not kept.any():
            continue
        given = model(rows[kept])
        for key, value in zip(
            (k for k, w in zip(mine, kept, strict=True) if w), given, strict=True
        ):
            out[key] = float(value)


def xgboost_fit(jobs: int) -> Fit:  # pragma: no cover - needs XGBoost
    import xgboost as xgb

    def fit(rows: Floats, targets: Floats, seed: int) -> Callable[[Floats], Floats]:
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=jobs,
            random_state=seed,
        )
        model.fit(rows, targets)

        def answer(new: Floats) -> Floats:
            out: Floats = model.predict(new).astype(np.float64)
            return out

        return answer

    return fit


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--farm", required=True)
    parser.add_argument("--hourly", type=Path, required=True, help="telemetry_hourly_means's")
    parser.add_argument("--cells", type=Path, required=True, help="fleet_cold_start_cells's")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    arguments = parser.parse_args(argv)
    if "torch" in sys.modules:
        raise SystemExit("torch is loaded: XGBoost's OpenMP must not share its process")

    turbines = loaded_farm(arguments.hourly)
    files = sorted(arguments.cells.glob(f"answers-{arguments.farm}-N*.csv"))
    by_file = {path: read_cells(path) for path in files}
    every = [k for keys in by_file.values() for k in keys]
    out = answered(
        turbines,
        every,
        xgboost_fit(arguments.jobs),
        seed=arguments.seed,
        say=lambda text: print(text, file=sys.stderr, flush=True),
    )
    for path, keys in by_file.items():
        label = path.stem.removeprefix("answers-")
        write(arguments.cells / f"trees-{label}.csv", keys, out, tuple(TREES))
        print(f"{label}: {len(keys)} cells")


if __name__ == "__main__":  # pragma: no cover
    main()
