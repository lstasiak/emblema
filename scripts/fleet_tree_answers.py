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

The fine-tuned forecaster's pilot reads trees on the same footing as its fits, under answers of
their own names, written as ``<kind>-<farm>-N<days>.csv``:

- ``--fit-until DAY``: every tree fitted on the hours before that day, whatever the start's year
  (``_until<year>``);
- ``--folds K``: the new turbine's whole fold left out with it, as ``fleet_adaptation.folds``
  deals the farm (``_folds``);
- ``--fleet-from DIR``: fitted on the other farm's turbines, every one, instead of the
  farm's own (``other_farm_`` for ``fleet_``).

The trees are handed the tuning the forecaster's recipe is, each setting an answer of its own,
named by ``--variant`` (``_<variant>``): more rounds at a lower rate (``--rounds``,
``--rate``), every row or another most (``--most-rows``, nought for every row), the rows whose
operating conditions are not all read, which the forecaster's fits read too (``--gaps``), and
the other farm's turbines beside the farm's own (``--fleet-beside DIR``). With ``--history``
each tree answer is also written corrected by the new turbine's own hours since its start
(``fleet_history_answers.py``: ``_offset``, ``_ridge``, ``_boosted``).
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
from scripts.fleet_adaptation import folds  # noqa: E402
from scripts.fleet_cold_start_cells import (  # noqa: E402
    FLEET,
    FLEET_NBM,
    FLEET_NBM_CURRENTS,
    FLEET_ROWS,
    Rows,
    latest,
    scorable,
)
from scripts.fleet_history_answers import SUFFIXES, corrected, history_of  # noqa: E402
from scripts.fleet_history_answers import Fit as Fit  # noqa: E402
from scripts.fleet_hours import (  # noqa: E402
    TARGETS,
    Turbine,
    another_farm,
    hour_of,
    loaded_farm,
    parsed_axis,
)

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


def year_start(day: date) -> int:
    return hour_of(date(day.year, 1, 1))


def fleet_rows(
    turbines: Sequence[Turbine],
    read: Mapping[str, tuple[Mask, Floats]],
    kind: Rows,
    channel: int,
    until: int,
    *,
    gaps: bool = False,
) -> dict[str, tuple[Floats, Floats]]:
    """Per turbine, its rows of the channel before ``until`` and their targets.

    An hour is kept where the channel is scored and every reading the rows need is there;
    with ``gaps``, wherever the channel is scored, a reading the rows lack left missing.
    ``read`` holds each turbine's ``scorable`` and ``latest``, which read no hour after their
    own, so one pass serves every year.
    """
    out = {}
    for turbine in turbines:
        scored, recent = read[turbine.name]
        hours = np.flatnonzero(scored[:until, channel])
        rows = kind.of(turbine.values, recent, hours, channel)
        kept = np.ones(len(rows), dtype=np.bool_) if gaps else np.isfinite(rows).all(axis=1)
        out[turbine.name] = (rows[kept], turbine.values[hours[kept], channel])
    return out


def drawn(rows: Floats, targets: Floats, seed: int, most: int = MOST_ROWS) -> tuple[Floats, Floats]:
    """At most ``most`` of the rows, drawn by ``seed``, in their order; every row at nought."""
    if not most or len(rows) <= most:
        return rows, targets
    keep = np.sort(np.random.default_rng(seed).choice(len(rows), most, replace=False))
    return rows[keep], targets[keep]


def answered(
    turbines: Sequence[Turbine],
    cells: Sequence[Key],
    fit: Fit,
    *,
    seed: int,
    cut: date | None = None,
    fleet: Sequence[Turbine] | None = None,
    left_out: Mapping[str, Sequence[str]] | None = None,
    trees: Sequence[str] = tuple(TREES),
    beside: Sequence[Turbine] = (),
    most: int = MOST_ROWS,
    gaps: bool = False,
    small: Fit | None = None,
    say: Callable[[str], None] = lambda _: None,
) -> dict[str, dict[Key, float]]:
    """Every tree answer for every cell it can answer.

    The trees read the farm's other turbines before the start's year; before ``cut`` instead,
    when given; ``fleet``'s turbines instead of the farm's, when given, and ``beside``'s with
    them; and none that ``left_out`` names beside the new turbine. A fit reads at most ``most``
    rows, and with ``gaps`` those too whose conditions are not all read. With ``small``, the
    fit of the small trees, each answer is also given corrected by the new turbine's own hours,
    under ``<tree><suffix>`` for each of ``fleet_history_answers.SUFFIXES``.

    Raises:
        ValueError: If a cell names a turbine not loaded, or starts before the cut, or two
            turbines the trees read share a name.
    """
    by_name = {t.name: t for t in turbines}
    groups: dict[tuple[str, int], list[Key]] = {}
    for key in cells:
        name, day = parsed_axis(key[1])
        if name not in by_name:
            raise ValueError(f"{key[1]} names a turbine not loaded")
        if cut is not None and day < cut:
            raise ValueError(f"{key[1]} starts before the trees' cut {cut}")
        groups.setdefault((name, hour_of(cut) if cut else year_start(day)), []).append(key)
    source = [*(turbines if fleet is None else fleet), *beside]
    if len({t.name for t in source}) != len(source):
        raise ValueError("two turbines the trees read share a name")
    suffixes = ("", *SUFFIXES) if small else ("",)
    out: dict[str, dict[Key, float]] = {tree + s: {} for tree in trees for s in suffixes}
    read = {t.name: (scorable(t.values), latest(t.values)) for t in source}
    started = time.monotonic()
    for until in sorted({u for _, u in groups}):
        for tree in trees:
            kind = ROWS[tree]
            for channel in range(len(TARGETS)):
                rows_of = fleet_rows(source, read, kind, channel, until, gaps=gaps)
                for (name, year), keys in groups.items():
                    if year != until:
                        continue
                    wanted = [k for k in keys if k[2] == channel]
                    unread = {name, *(left_out or {}).get(name, ())}
                    others = [v for t, v in rows_of.items() if t not in unread]
                    if not wanted or not others:
                        continue
                    rows = np.concatenate([r for r, _ in others])
                    if len(rows) < ROWS_PER_FEATURE * rows.shape[1]:
                        continue
                    targets = np.concatenate([y for _, y in others])
                    model = fit(*drawn(rows, targets, seed + channel, most), seed)
                    into = {s: out[tree + s] for s in suffixes}
                    answer_cells(
                        into, by_name[name], kind, channel, wanted, model, small=small, seed=seed
                    )
            say(
                f"{tree} before {until}: {len(out[tree])} cells, {time.monotonic() - started:.0f} s"
            )
    return out


def answer_cells(
    out: Mapping[str, dict[Key, float]],
    turbine: Turbine,
    kind: Rows,
    channel: int,
    keys: Sequence[Key],
    model: Callable[[Floats], Floats],
    *,
    small: Fit | None = None,
    seed: int = 0,
) -> None:
    """The model's answer for each of the new turbine's cells whose rows are whole.

    ``out`` holds the answers by suffix: the model's own under the empty one and, with
    ``small``, each correction by the turbine's own hours before the cell.
    """
    for axis in sorted({k[1] for k in keys}):
        start = hour_of(parsed_axis(axis)[1])
        new = turbine.before(start)
        mine = [k for k in keys if k[1] == axis]
        hours = np.asarray([k[0] for k in mine], dtype=np.int64)
        recent = latest(new.values)
        rows = kind.of(new.values, recent, hours, channel)
        kept = np.isfinite(rows).all(axis=1)
        if not kept.any():
            continue
        hours, rows = hours[kept], rows[kept]
        given = {"": model(rows)}
        if small:
            end = int(hours.max())
            own = history_of(new.values, recent, kind, channel, start, end, model)
            conditions = new.values[hours][:, list(kind.conditions)]
            given |= corrected(
                own, hours, rows, given[""], conditions, start=start, small=small, seed=seed
            )
        answered_keys = [k for k, w in zip(mine, kept, strict=True) if w]
        for suffix, values in given.items():
            for key, value in zip(answered_keys, values, strict=True):
                out[suffix][key] = float(value)


def named_as(
    tree: str, *, cut: date | None, folds: bool, other: bool, variant: str = "", suffix: str = ""
) -> str:
    """A tree answer's name when its trees read otherwise than the farm's before the year."""
    name = tree.replace("fleet_", "other_farm_", 1) if other else tree
    name += ("_folds" if folds else "") + (f"_until{cut.year}" if cut else "")
    return name + (f"_{variant}" if variant else "") + suffix


def other_farm(directory: Path, own: Path) -> list[Turbine]:  # pragma: no cover - reads the data
    """Another farm's turbines, named under their directory so that none shares a name."""
    return [Turbine(f"{directory.name}/{t.name}", t.values) for t in another_farm(directory, own)]


def small_fit() -> Fit:  # pragma: no cover - needs XGBoost
    """The small trees fitted to a new turbine's own residuals.

    They read several hundred rows of a first month, so they are shallow and slow, each round
    reads a share of the rows, and 32 bins hold a feature; one thread, as a fit takes
    milliseconds.
    """
    return xgboost_fit(1, rounds=100, rate=0.05, max_depth=3, subsample=0.8, max_bin=32)


def xgboost_fit(
    jobs: int, *, rounds: int = 300, rate: float = 0.1, **settings: float
) -> Fit:  # pragma: no cover - needs XGBoost
    import xgboost as xgb

    def fit(rows: Floats, targets: Floats, seed: int) -> Callable[[Floats], Floats]:
        model = xgb.XGBRegressor(
            n_estimators=rounds,
            learning_rate=rate,
            tree_method="hist",
            n_jobs=jobs,
            random_state=seed,
            **settings,
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
    parser.add_argument("--fit-until", type=date.fromisoformat, help="the first day not read")
    parser.add_argument("--folds", type=int, help="leave the new turbine's fold out")
    parser.add_argument(
        "--fleet-from", type=Path, metavar="DIR", help="the other farm's hourly means"
    )
    parser.add_argument("--fleet-beside", type=Path, metavar="DIR", help="read it as well")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--most-rows", type=int, default=MOST_ROWS, help="0: every row")
    parser.add_argument("--gaps", action="store_true", help="fit rows lacking a condition too")
    parser.add_argument("--variant", default="", help="names the answers of trees tuned otherwise")
    parser.add_argument("--history", action="store_true", help="corrected by own hours too")
    parser.add_argument("--trees", nargs="+", choices=tuple(TREES), default=tuple(TREES))
    parser.add_argument("--kind", default="trees", help="the written files' prefix")
    arguments = parser.parse_args(argv)
    if "torch" in sys.modules:
        raise SystemExit("torch is loaded: XGBoost's OpenMP must not share its process")
    tuned = (
        arguments.rounds,
        arguments.rate,
        arguments.most_rows,
        arguments.gaps,
        arguments.fleet_beside,
    ) != (300, 0.1, MOST_ROWS, False, None)
    if tuned != bool(arguments.variant):
        raise SystemExit("trees tuned otherwise are named by --variant, and no others")
    varied = (arguments.fit_until, arguments.folds, arguments.fleet_from) != (None, None, None)
    if (varied or tuned or arguments.history) and arguments.kind == "trees":
        raise SystemExit("trees read otherwise are written under a --kind of their own")

    turbines = loaded_farm(arguments.hourly)
    fleet = other_farm(arguments.fleet_from, arguments.hourly) if arguments.fleet_from else None
    left_out = None
    if arguments.folds is not None:
        dealt = folds([t.name for t in turbines], arguments.folds)
        left_out = {name: fold for fold in dealt for name in fold}
    files = sorted(arguments.cells.glob(f"answers-{arguments.farm}-N*.csv"))
    by_file = {path: read_cells(path) for path in files}
    every = [k for keys in by_file.values() for k in keys]
    out = answered(
        turbines,
        every,
        xgboost_fit(arguments.jobs, rounds=arguments.rounds, rate=arguments.rate),
        seed=arguments.seed,
        cut=arguments.fit_until,
        fleet=fleet,
        left_out=left_out,
        trees=arguments.trees,
        beside=(
            other_farm(arguments.fleet_beside, arguments.hourly) if arguments.fleet_beside else ()
        ),
        most=arguments.most_rows,
        gaps=arguments.gaps,
        small=small_fit() if arguments.history else None,
        say=lambda text: print(text, file=sys.stderr, flush=True),
    )
    renamed = {
        named_as(
            tree,
            cut=arguments.fit_until,
            folds=arguments.folds is not None,
            other=bool(arguments.fleet_from),
            variant=arguments.variant,
            suffix=suffix,
        ): out[tree + suffix]
        for tree in arguments.trees
        for suffix in (("", *SUFFIXES) if arguments.history else ("",))
    }
    for path, keys in by_file.items():
        label = path.stem.removeprefix("answers-")
        write(arguments.cells / f"{arguments.kind}-{label}.csv", keys, renamed, tuple(renamed))
        print(f"{label}: {len(keys)} cells")


if __name__ == "__main__":  # pragma: no cover
    main()
