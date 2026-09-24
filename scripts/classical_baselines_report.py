"""MiniRocket as implemented here against the reference implementation, aeon's.

Two measurements, each written to CSV before anything is printed from it:

- ``agreement.csv``: this project's MiniRocket against aeon's where no draw differs — one
  channel and one fitted series, or many channels fitted under aeon's own random draws —
  feature by feature;
- ``multivariate.csv``: three fits under the same grid, scaling and ridge — aeon's, this
  project's under its own draws, and this project's under aeon's draws — fitted on a budget of
  windows of a published corpus and scored on units held out, per seed. The target is the share
  of its unit's run a window ends at, a remaining-life shape read off the block alone, so the
  comparison needs no registry and no labels file. The third fit tells a difference of
  arithmetic from a difference of draws: equal errors under equal draws leave only the draws.

What a cell of every classical method costs is read off the campaigns that ran them, which time
every cell where it runs. aeon is not a dependency of the project, so the measurement is run
under it::

    uv run --with aeon python scripts/classical_baselines_report.py --block <path> --part aeon
    uv run python scripts/classical_baselines_report.py --block <path> --part print
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid  # noqa: E402
from emblema.evaluation.adapters.minirocket.minirocket_transform import (  # noqa: E402
    MiniRocketTransform,
)
from emblema.shared.adapters.windows.window_block import WindowBlock  # noqa: E402
from scripts.reporting import dated_heading, machine, table  # noqa: E402

OUT = REPO_ROOT / "data" / "report" / "classical-baselines"
# The configuration the environment template states, so the cost is the cost of what runs.
FEATURES, GRID_STEPS, THREADS = 9996, 64, 4
PENALTIES = (0.001, 0.00464, 0.0215, 0.1, 0.464, 2.15, 10.0, 46.4, 215.0, 1000.0)
# Enough repeats that a gap between two implementations can be told from a lucky draw.
SEEDS = tuple(range(1, 21))
# This project's fit replaying aeon's random draws, which leaves only the arithmetic to differ.
REPLAYED = "emblema under aeon's draws"
MULTIVARIATE_BUDGETS = (200, 1000)
HELD_OUT_SHARE = 0.3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", type=Path, required=True, help="a published corpus block")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--part", choices=("aeon", "print"), required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.part == "aeon":
        write(arguments.out / "agreement.csv", agreement())
        write(arguments.out / "multivariate.csv", multivariate(WindowBlock(arguments.block)))
    else:
        report(arguments.out, arguments.block)


class AeonDraws:
    """aeon's random draws, offered through the calls this project's fit draws with.

    aeon seeds NumPy's legacy generator for the channels and seeds numba's again for the series
    each bias is read off; replaying both streams makes the one remaining difference between
    the two fits the arithmetic, which is what the comparison is for.
    """

    def __init__(self, seed: int) -> None:
        self._combinations = np.random.RandomState(seed)
        self._examples = np.random.RandomState(seed)

    def uniform(self, low: float, high: float, size: int) -> NDArray[np.float64]:
        return self._combinations.uniform(low, high, size)

    def choice(self, n: int, size: int, *, replace: bool) -> NDArray[np.int64]:
        return self._combinations.choice(n, size, replace=replace)

    def integers(self, n: int, *, size: int) -> NDArray[np.int64]:
        return np.array([self._examples.randint(n) for _ in range(size)])


def agreement() -> list[dict[str, str]]:
    """Where no draw differs, the two transforms must agree to the precision aeon computes in."""
    from aeon.transformations.collection.convolution_based import (  # ty: ignore[unresolved-import]
        MiniRocket,
    )

    rows = []
    for length, features in ((64, 840), (128, 9996), (40, 84)):
        rng = np.random.default_rng(length)
        fitted = rng.normal(size=(1, 1, length)).astype(np.float32).astype(np.float64)
        scored = rng.normal(size=(20, 1, length)).astype(np.float32).astype(np.float64)
        theirs = MiniRocket(n_kernels=features, random_state=1).fit(fitted).transform(scored)
        ours = MiniRocketTransform.fitted(fitted, features, np.random.default_rng(1)).of(scored)
        rows.append(difference("one channel, one series", length, features, theirs, ours))
    rng = np.random.default_rng(9)
    fitted = rng.normal(size=(40, 42, 64)).astype(np.float32).astype(np.float64)
    scored = rng.normal(size=(20, 42, 64)).astype(np.float32).astype(np.float64)
    for seed in (1, 2, 3):
        theirs = MiniRocket(n_kernels=FEATURES, random_state=seed).fit(fitted).transform(scored)
        replayed = MiniRocketTransform.fitted(fitted, FEATURES, AeonDraws(seed))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        rows.append(
            difference(
                f"42 channels, aeon's draws, seed {seed}", 64, FEATURES, theirs, replayed.of(scored)
            )
        )
    return rows


def difference(
    case: str, length: int, features: int, theirs: object, ours: NDArray[np.float64]
) -> dict[str, str]:
    """How far apart two feature matrices are: the largest gap, and the share of cells apart.

    A cell apart is a share of positions that differs by at least one position: a convolution
    that lands exactly on a bias in one precision and just past it in the other.
    """
    apart = np.abs(np.asarray(theirs, dtype=np.float64) - ours)
    return {
        "case": case,
        "length": str(length),
        "features": str(features),
        "max_abs_difference": repr(float(apart.max())),
        "cells_apart": repr(float((apart > 0.5 / length).mean())),
    }


def multivariate(block: WindowBlock) -> list[dict[str, str]]:
    from aeon.transformations.collection.convolution_based import (  # ty: ignore[unresolved-import]
        MiniRocket,
    )
    from sklearn.linear_model import RidgeCV
    from threadpoolctl import threadpool_limits

    units = np.array([block.unit_of(index) for index in range(len(block))])
    ends = np.array([block.extent_of(index)[1] for index in range(len(block))])
    lives = {unit: ends[units == unit].max() for unit in np.unique(units)}
    target = np.array([end / lives[unit] for unit, end in zip(units, ends, strict=True)])
    ranked = np.random.default_rng(0).permutation(np.unique(units))
    held = set(ranked[: int(len(ranked) * HELD_OUT_SHARE)].tolist())
    training = np.flatnonzero([unit not in held for unit in units])
    testing = np.random.default_rng(0).choice(
        np.flatnonzero([unit in held for unit in units]), 2000, replace=False
    )
    channels = max(max(window.channel_ids) for window in block.at(range(0, len(block), 97)))
    grid = RegularGrid(GRID_STEPS, channels)
    tested = grid.of(block.at(sorted(testing.tolist())))
    truth = target[sorted(testing.tolist())]
    rows = []
    for budget in MULTIVARIATE_BUDGETS:
        for seed in SEEDS:
            drawn = sorted(np.random.default_rng(seed).choice(training, budget, replace=False))
            laid = grid.of(block.at(drawn))
            for implementation in ("emblema", "aeon", REPLAYED):
                started = time.perf_counter()
                with threadpool_limits(limits=THREADS):
                    if implementation != "aeon":
                        draws = (
                            np.random.default_rng(seed)
                            if implementation == "emblema"
                            else AeonDraws(seed)
                        )
                        transform = MiniRocketTransform.fitted(laid, FEATURES, draws)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                        fitted, scored = transform.of(laid), transform.of(tested)
                    else:
                        theirs = MiniRocket(n_kernels=FEATURES, random_state=seed).fit(laid)
                        fitted = np.asarray(theirs.transform(laid))
                        scored = np.asarray(theirs.transform(tested))
                    spread = fitted.std(axis=0)
                    scale = np.where(spread > 0.0, spread, 1.0)
                    ridge = RidgeCV(alphas=PENALTIES).fit(fitted / scale, target[drawn])
                    answer = ridge.predict(scored / scale)
                rows.append(
                    {
                        "implementation": implementation,
                        "budget": str(budget),
                        "seed": str(seed),
                        "rmse": repr(float(np.sqrt(np.mean((answer - truth) ** 2)))),
                        "penalty": repr(float(ridge.alpha_)),
                        "seconds": repr(time.perf_counter() - started),
                    }
                )
    return rows


def write(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def report(out: Path, block: Path) -> None:
    print(dated_heading())
    print(f"\n{machine()}; block `{block.name[:12]}…`, grid {GRID_STEPS}, {FEATURES} features.\n")
    print(
        table(
            ("case", "length", "features", "largest difference", "cells apart"),
            (
                (
                    r["case"],
                    r["length"],
                    r["features"],
                    f"{float(r['max_abs_difference']):.1e}",
                    f"{float(r['cells_apart']):.1e}",
                )
                for r in read(out / "agreement.csv")
            ),
        )
    )
    measured = read(out / "multivariate.csv")
    print()
    print(
        table(
            (
                "budget",
                "emblema RMSE",
                "aeon RMSE",
                "paired gap",
                "emblema lower in",
                "under aeon's draws",
                "emblema s",
                "aeon s",
            ),
            (summary(measured, budget) for budget in MULTIVARIATE_BUDGETS),
        )
    )


def summary(rows: Sequence[dict[str, str]], budget: int) -> tuple[str, ...]:
    """Both implementations over the seeds, paired seed by seed on the same drawn windows."""

    def of(implementation: str, field: str) -> NDArray[np.float64]:
        by_seed = {
            r["seed"]: float(r[field])
            for r in rows
            if r["implementation"] == implementation and r["budget"] == str(budget)
        }
        return np.array([by_seed[seed] for seed in sorted(by_seed, key=int)])

    ours, theirs = of("emblema", "rmse"), of("aeon", "rmse")
    gap = ours - theirs
    return (
        str(budget),
        f"{ours.mean():.4f} ± {ours.std(ddof=1):.4f}",
        f"{theirs.mean():.4f} ± {theirs.std(ddof=1):.4f}",
        f"{gap.mean():+.4f} ± {gap.std(ddof=1):.4f}",
        f"{int((gap < 0).sum())} of {len(gap)}",
        f"{np.abs(of(REPLAYED, 'rmse') - theirs).max():.1e} apart at most",
        f"{of('emblema', 'seconds').mean():.1f}",
        f"{of('aeon', 'seconds').mean():.1f}",
    )


if __name__ == "__main__":
    main()
