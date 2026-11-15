r"""The forecasting pretext read against the ridge and the answers beside it, from the cells.

Reads the per-cell files the answers report wrote for each pretraining and set of months, and
the published forecaster's file where there is one, and prints the declared tables: every
answer's loss on each set and mission with 95 % intervals from resampling whole windows, and the
verdict's reading, the decoder's loss over the ridge's on the same cells. The arm is the mean of
the pretrainings' ratios; its threshold is twice the root of the resampling variance of that mean
plus half the pretrainings' spread squared over their count, the spread floored, as the stage
reads an arm of a few pretrainings against a bar that has no spread of its own. The level is
read off the arm's interval against one and against the least gain: a ratio whose interval
covers one is less than about the threshold either way, not none. Beside it, descriptively: the
decoder against the answers with no model and against the ridge that also reads the window's
other channels, the decoder and the ridge against the published forecaster on the mean's squared
error, and the median over windows of the decoder's loss over the ridge's within each window,
which a bar's failure on a few windows cannot carry.

    uv run scripts/history_forecast_reading.py \\
        --answers seed1 unread DIR/seed1/answers-unread.csv \\
        --answers seed2 unread DIR/seed2/answers-unread.csv \\
        [--chronos unread DIR/chronos2-unread.csv] \\
        --least-gain 0.8 --floor 0.02 [--delta 1.0] [--resamples 10000] --out DIR

Writes ``losses.csv`` and ``verdicts.csv`` under ``--out`` and prints both as tables.
"""

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ANSWERS = ("decoder", "last_reading", "median_of_history", "same_phase", "ridge", "window_ridge")
PUBLISHED = "chronos2"
MEASURES = ("huber_five", "huber_mean", "squared_mean")
LOSS_COLUMNS = (
    "set",
    "axis",
    "pretraining",
    "answer",
    "measure",
    "windows",
    "cells",
    "mean",
    "low",
    "high",
)
VERDICT_COLUMNS = (
    "set",
    "axis",
    "pretraining",
    "measure",
    "of",
    "against",
    "ratio",
    "low",
    "high",
    "spread",
    "threshold",
    "level",
)

Key = tuple[int, str, int]
Floats = NDArray[np.float64]


@dataclass(frozen=True)
class CellRow:
    """One scored cell: its target's mean, and each answer's Huber loss and mean."""

    window: int
    axis: str
    channel: int
    target_mean: float
    huber: Mapping[str, float]
    means: Mapping[str, float]

    @property
    def key(self) -> Key:
        return (self.window, self.axis, self.channel)


def read_answers(path: Path) -> list[CellRow]:
    with path.open(newline="") as handle:
        return [
            CellRow(
                window=int(row["window"]),
                axis=row["axis"],
                channel=int(row["channel"]),
                target_mean=float(row["target_mean"]),
                huber={a: float(row[f"huber_{a}"]) for a in ANSWERS},
                means={a: float(row[f"mean_{a}"]) for a in ANSWERS},
            )
            for row in csv.DictReader(handle)
        ]


def read_published(path: Path) -> dict[Key, float]:
    with path.open(newline="") as handle:
        return {
            (int(row["window"]), row["axis"], int(row["channel"])): float(row[f"mean_{PUBLISHED}"])
            for row in csv.DictReader(handle)
        }


def huber(error: Floats, delta: float) -> Floats:
    magnitude = np.abs(error)
    return np.where(magnitude <= delta, 0.5 * error**2, delta * (magnitude - 0.5 * delta))


@dataclass(frozen=True)
class CellSet:
    """The cells of one set and axis, every answer's losses on them, laid for resampling.

    Attributes:
        windows: Sorted identifiers of the windows the cells lie in.
        inverse: For each cell, the position of its window in ``windows``.
        losses: ``{(answer, measure): [cells]}``.
    """

    windows: NDArray[np.int64]
    inverse: NDArray[np.intp]
    losses: Mapping[tuple[str, str], Floats]

    @classmethod
    def of(
        cls, cells: Sequence[CellRow], *, delta: float, published: Mapping[Key, float] | None
    ) -> "CellSet":
        """The cells laid out, the published answer joined by cell where given.

        Raises:
            ValueError: If no cell is given, or the published answer misses one of them.
        """
        if not cells:
            raise ValueError("no cell to read")
        windows, inverse = np.unique([c.window for c in cells], return_inverse=True)
        target = np.array([c.target_mean for c in cells])
        losses: dict[tuple[str, str], Floats] = {}
        answers: dict[str, Floats] = {a: np.array([c.means[a] for c in cells]) for a in ANSWERS}
        for a in ANSWERS:
            losses[(a, "huber_five")] = np.array([c.huber[a] for c in cells])
        if published is not None:
            missing = [c.key for c in cells if c.key not in published]
            if missing:
                raise ValueError(f"the published answer misses {len(missing)} cells: {missing[0]}")
            answers[PUBLISHED] = np.array([published[c.key] for c in cells])
        for a, mean in answers.items():
            losses[(a, "huber_mean")] = huber(mean - target, delta)
            losses[(a, "squared_mean")] = (mean - target) ** 2
        return cls(windows.astype(np.int64), inverse.astype(np.intp), losses)

    @property
    def cells(self) -> int:
        return len(self.inverse)

    def per_window(self, values: Floats) -> tuple[Floats, Floats]:
        sums = np.zeros(len(self.windows))
        np.add.at(sums, self.inverse, values)
        counts = np.bincount(self.inverse, minlength=len(self.windows)).astype(np.float64)
        return sums, counts

    def resampled_means(self, values: Floats, draws: NDArray[np.int64]) -> tuple[float, Floats]:
        """The cells' mean and its value under each draw of windows."""
        sums, counts = self.per_window(values)
        point = float(sums.sum() / counts.sum())
        boot: Floats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return point, boot


def draws_of(windows: int, resamples: int, seed: int) -> NDArray[np.int64]:
    return np.random.default_rng(seed).integers(0, windows, (resamples, windows), dtype=np.int64)


def interval(boot: Floats) -> tuple[float, float]:
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def level_of(low: float, high: float, least_gain: float) -> str:
    """What the arm's interval around one and the least gain says, a ratio below one a gain."""
    if low > 1.0:
        return "harm"
    if high >= 1.0:
        return "noise: less than about the threshold either way"
    if high <= least_gain:
        return "at least the least gain"
    return "real but small"


def threshold(boot_arm: Floats, ratios: Sequence[float], floor: float) -> tuple[float, float]:
    """The pretrainings' spread floored and the arm's threshold, as the stage sets them."""
    spread = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    spread = max(floor, spread)
    se = float(np.std(boot_arm, ddof=1))
    return spread, 2.0 * float(np.sqrt(se**2 + (spread**2 / 2.0) / len(ratios)))


def losses_table(
    sets: Mapping[tuple[str, str, str], CellSet],
    draws: Mapping[tuple[str, str], NDArray[np.int64]],
) -> list[dict[str, object]]:
    """Every answer's mean loss by set, axis and pretraining, under each measure."""
    rows: list[dict[str, object]] = []
    for (name, axis, pretraining), cells in sorted(sets.items()):
        for (answer, measure), values in cells.losses.items():
            point, boot = cells.resampled_means(values, draws[(name, axis)])
            low, high = interval(boot)
            rows.append(
                {
                    "set": name,
                    "axis": axis,
                    "pretraining": pretraining,
                    "answer": answer,
                    "measure": measure,
                    "windows": len(cells.windows),
                    "cells": cells.cells,
                    "mean": point,
                    "low": low,
                    "high": high,
                }
            )
    return rows


def ratio_rows(
    sets: Mapping[tuple[str, str, str], CellSet],
    draws: Mapping[tuple[str, str], NDArray[np.int64]],
    *,
    numerator: str,
    against: str,
    measure: str,
    least_gain: float,
    floor: float,
    verdict: bool = False,
) -> list[dict[str, object]]:
    """``numerator`` over ``against`` per pretraining and for the arm, by set and axis.

    Only the comparison declared as the verdict reads a level; the others are descriptive.
    """
    rows: list[dict[str, object]] = []
    by_set: dict[tuple[str, str], list[tuple[str, CellSet]]] = {}
    for (name, axis, pretraining), cells in sorted(sets.items()):
        by_set.setdefault((name, axis), []).append((pretraining, cells))
    for (name, axis), members in by_set.items():
        ratios: list[float] = []
        boots: list[Floats] = []
        for pretraining, cells in members:
            if (against, measure) not in cells.losses:
                continue
            top, boot_top = cells.resampled_means(
                cells.losses[(numerator, measure)], draws[(name, axis)]
            )
            bottom, boot_bottom = cells.resampled_means(
                cells.losses[(against, measure)], draws[(name, axis)]
            )
            ratio, boot = top / bottom, boot_top / boot_bottom
            ratios.append(ratio)
            boots.append(boot)
            low, high = interval(boot)
            rows.append(
                {
                    "set": name,
                    "axis": axis,
                    "pretraining": pretraining,
                    "measure": measure,
                    "of": numerator,
                    "against": against,
                    "ratio": ratio,
                    "low": low,
                    "high": high,
                    "spread": "",
                    "threshold": "",
                    "level": "",
                }
            )
        if not ratios:
            continue
        arm = float(np.mean(ratios))
        boot_arm = np.mean(boots, axis=0)
        spread, t = threshold(boot_arm, ratios, floor)
        rows.append(
            {
                "set": name,
                "axis": axis,
                "pretraining": "arm",
                "measure": measure,
                "of": numerator,
                "against": against,
                "ratio": arm,
                "low": arm - t,
                "high": arm + t,
                "spread": spread,
                "threshold": t,
                "level": level_of(arm - t, arm + t, least_gain) if verdict else "descriptive",
            }
        )
    return rows


def median_ratio_rows(
    sets: Mapping[tuple[str, str, str], CellSet],
    draws: Mapping[tuple[str, str], NDArray[np.int64]],
    *,
    numerator: str,
    against: str,
    measure: str,
) -> list[dict[str, object]]:
    """The median over windows of ``numerator`` over ``against`` within each window, descriptively.

    The verdict's ratio is of means, which a bar's failure on a few windows can carry; the
    median of the windows' own ratios says whether the typical window agrees.
    """
    rows: list[dict[str, object]] = []
    for (name, axis, pretraining), cells in sorted(sets.items()):
        top, _ = cells.per_window(cells.losses[(numerator, measure)])
        bottom, _ = cells.per_window(cells.losses[(against, measure)])
        if (bottom == 0).any():
            raise ValueError(f"{against} missed by nothing on a window of {name} {axis}: no ratio")
        per_window = top / bottom
        boot = np.median(per_window[draws[(name, axis)]], axis=1)
        low, high = interval(boot)
        rows.append(
            {
                "set": name,
                "axis": axis,
                "pretraining": pretraining,
                "measure": f"{measure}_window_median",
                "of": numerator,
                "against": against,
                "ratio": float(np.median(per_window)),
                "low": low,
                "high": high,
                "spread": "",
                "threshold": "",
                "level": "descriptive",
            }
        )
    return rows


def check_same_cells(sets: Mapping[tuple[str, str, str], CellSet]) -> None:
    """The pretrainings of one set and axis must have been scored on the same windows.

    Raises:
        ValueError: If they were not.
    """
    seen: dict[tuple[str, str], tuple[str, NDArray[np.int64]]] = {}
    for (name, axis, pretraining), cells in sorted(sets.items()):
        if (name, axis) in seen:
            other, windows = seen[(name, axis)]
            if not np.array_equal(windows, cells.windows):
                raise ValueError(
                    f"{pretraining} and {other} were scored on different windows of {name} {axis}"
                )
        else:
            seen[(name, axis)] = (pretraining, cells.windows)


def markdown(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    def shown(value: object) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(shown(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def write(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def read(
    answers: Sequence[tuple[str, str, Path]],
    published: Sequence[tuple[str, Path]],
    *,
    delta: float,
    least_gain: float,
    floor: float,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The two tables from the files: every loss, and every ratio with the arm's verdict."""
    published_by_set = {name: read_published(path) for name, path in published}
    sets: dict[tuple[str, str, str], CellSet] = {}
    for pretraining, name, path in answers:
        cells = read_answers(path)
        for axis in sorted({c.axis for c in cells}):
            sets[(name, axis, pretraining)] = CellSet.of(
                [c for c in cells if c.axis == axis],
                delta=delta,
                published=published_by_set.get(name),
            )
    check_same_cells(sets)
    draws = {
        (name, axis): draws_of(len(cells.windows), resamples, seed)
        for (name, axis, _), cells in sets.items()
    }
    verdicts = ratio_rows(
        sets,
        draws,
        numerator="decoder",
        against="ridge",
        measure="huber_five",
        least_gain=least_gain,
        floor=floor,
        verdict=True,
    )
    verdicts += median_ratio_rows(
        sets, draws, numerator="decoder", against="ridge", measure="huber_five"
    )
    for against in ("last_reading", "median_of_history", "same_phase", "window_ridge"):
        verdicts += ratio_rows(
            sets,
            draws,
            numerator="decoder",
            against=against,
            measure="huber_five",
            least_gain=least_gain,
            floor=floor,
        )
    for numerator in ("decoder", "ridge"):
        verdicts += ratio_rows(
            sets,
            draws,
            numerator=numerator,
            against=PUBLISHED,
            measure="squared_mean",
            least_gain=least_gain,
            floor=floor,
        )
    return losses_table(sets, draws), verdicts


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--answers", nargs=3, action="append", required=True, metavar=("PRETRAINING", "SET", "PATH")
    )
    parser.add_argument("--chronos", nargs=2, action="append", default=[], metavar=("SET", "PATH"))
    parser.add_argument("--least-gain", type=float, required=True)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    losses, verdicts = read(
        [(p, s, Path(path)) for p, s, path in arguments.answers],
        [(s, Path(path)) for s, path in arguments.chronos],
        delta=arguments.delta,
        least_gain=arguments.least_gain,
        floor=arguments.floor,
        resamples=arguments.resamples,
        seed=arguments.seed,
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "losses.csv", LOSS_COLUMNS, losses)
    write(arguments.out / "verdicts.csv", VERDICT_COLUMNS, verdicts)
    print(markdown(LOSS_COLUMNS, losses))
    print()
    print(markdown(VERDICT_COLUMNS, verdicts))


if __name__ == "__main__":  # pragma: no cover
    main()
