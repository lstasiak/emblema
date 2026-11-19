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

With ``--verdict-of NAME --verdict-set SET --bar NAME...`` the verdict is instead, on that set's
rows alone, a published forecaster against
the best of the classical answers named: the geometric mean over windows of its squared error of
the mean over each answer's, window by window, as forecasting benchmarks aggregate relative
errors, so a few windows of extreme values carry no more than their share; the best answer is
the one the forecaster leads least, taken in every draw of windows, so the bar is never weaker
than any of its answers. The forecaster has no pretrainings, so its level is read off the
percentile interval of that ratio. The ratio of mean losses stands beside it, and the decoder's
reading is then descriptive. Further answers come from any file of ``mean_<name>`` columns per
cell (``--chronos`` or ``--beside``), as the published and the local forecasters write them.

    uv run scripts/history_forecast_reading.py \\
        --answers seed1 unread DIR/seed1/answers-unread.csv \\
        --answers seed2 unread DIR/seed2/answers-unread.csv \\
        [--chronos unread DIR/chronos2-unread.csv] [--beside unread DIR/local-unread.csv] \\
        [--verdict-of chronos2 --verdict-set unread --bar ridge window_ridge auto_ets ...] \\
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


def read_published(path: Path) -> dict[str, dict[Key, float]]:
    """Every ``mean_<name>`` column of a file of answers per cell, by name."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = [c.removeprefix("mean_") for c in reader.fieldnames or () if c.startswith("mean_")]
        out: dict[str, dict[Key, float]] = {name: {} for name in names}
        for row in reader:
            key = (int(row["window"]), row["axis"], int(row["channel"]))
            for name in names:
                out[name][key] = float(row[f"mean_{name}"])
    return out


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
        cls,
        cells: Sequence[CellRow],
        *,
        delta: float,
        published: Mapping[str, Mapping[Key, float]] | None,
    ) -> "CellSet":
        """The cells laid out, the answers from other files joined by cell where given.

        Raises:
            ValueError: If no cell is given, or an answer from another file misses one of them,
                or names an answer the cells already hold, or any loss is not finite.
        """
        if not cells:
            raise ValueError("no cell to read")
        windows, inverse = np.unique([c.window for c in cells], return_inverse=True)
        target = np.array([c.target_mean for c in cells])
        losses: dict[tuple[str, str], Floats] = {}
        answers: dict[str, Floats] = {a: np.array([c.means[a] for c in cells]) for a in ANSWERS}
        for a in ANSWERS:
            losses[(a, "huber_five")] = np.array([c.huber[a] for c in cells])
        for name, by_cell in (published or {}).items():
            if name in answers:
                raise ValueError(f"{name} is already an answer of the cells")
            missing = [c.key for c in cells if c.key not in by_cell]
            if missing:
                raise ValueError(f"{name} misses {len(missing)} cells: {missing[0]}")
            answers[name] = np.array([by_cell[c.key] for c in cells])
        for a, mean in answers.items():
            losses[(a, "huber_mean")] = huber(mean - target, delta)
            losses[(a, "squared_mean")] = (mean - target) ** 2
        for (a, measure), values in losses.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{a} holds a {measure} that is not finite")
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
    """What the arm's interval around one and the least gain says, a ratio below one a gain.

    Raises:
        ValueError: If an edge is not finite, which reads no level.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"an interval [{low}; {high}] reads no level")
    if low > 1.0:
        return "harm"
    if high >= 1.0:
        return "noise: less than about the threshold either way"
    if high <= least_gain:
        return "at least the least gain"
    return "real and small"


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


def window_log_ratios(cells: CellSet, numerator: str, against: str, measure: str) -> Floats:
    """Each window's log of ``numerator``'s mean loss over ``against``'s.

    Raises:
        ValueError: If either misses by nothing on a window, which leaves no ratio.
    """
    top, counts = cells.per_window(cells.losses[(numerator, measure)])
    bottom, _ = cells.per_window(cells.losses[(against, measure)])
    if (top == 0).any() or (bottom == 0).any():
        raise ValueError(f"{numerator} or {against} missed by nothing on a window: no ratio")
    logs: Floats = np.log((top / counts) / (bottom / counts))
    return logs


def best_bar_rows(
    sets: Mapping[tuple[str, str, str], CellSet],
    draws: Mapping[tuple[str, str], NDArray[np.int64]],
    *,
    numerator: str,
    bar: Sequence[str],
    measure: str,
    least_gain: float,
    verdict_set: str | None,
    geometric: bool,
) -> list[dict[str, object]]:
    """``numerator`` over the best of ``bar`` by set and axis, the best taken in every draw.

    With ``geometric`` the ratio is the geometric mean over windows of the windows' own mean
    losses' ratio, the best answer the one ``numerator`` leads least; without it, the ratio of
    mean losses. Only the rows of ``verdict_set`` read a level. The answers compared are the same
    under every pretraining, so the first one's cells are read.

    Raises:
        ValueError: If, read geometrically, an answer misses by nothing on a window.
    """
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for (name, axis, _), cells in sorted(sets.items()):
        if (name, axis) in seen:
            continue
        seen.add((name, axis))
        drawn = draws[(name, axis)]
        if geometric:
            logs = np.stack([window_log_ratios(cells, numerator, a, measure) for a in bar])
            points = np.exp(logs.mean(axis=1))
            best = int(np.argmax(points))
            ratio = float(points[best])
            low, high = interval(np.exp(np.max(logs[:, drawn].mean(axis=2), axis=0)))
        else:
            top, boot_top = cells.resampled_means(cells.losses[(numerator, measure)], drawn)
            bottoms = [cells.resampled_means(cells.losses[(a, measure)], drawn) for a in bar]
            best = int(np.argmin([point for point, _ in bottoms]))
            boot_best = np.min(np.stack([boot for _, boot in bottoms]), axis=0)
            ratio = top / bottoms[best][0]
            low, high = interval(boot_top / boot_best)
        rows.append(
            {
                "set": name,
                "axis": axis,
                "pretraining": "none",
                "measure": f"{measure}_window_geometric" if geometric else measure,
                "of": numerator,
                "against": bar[best] if len(bar) == 1 else f"best of {len(bar)}: {bar[best]}",
                "ratio": ratio,
                "low": low,
                "high": high,
                "spread": "",
                "threshold": "",
                "level": level_of(low, high, least_gain) if name == verdict_set else "descriptive",
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
    verdict_of: str | None = None,
    bar: Sequence[str] = (),
    verdict_set: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The two tables from the files: every loss, and every ratio with the verdict's level.

    Raises:
        ValueError: If a verdict of a published forecaster names no bar or no set, or the bar
            names it, or a bar or set is named without such a verdict.
    """
    if verdict_of is None and (bar or verdict_set is not None):
        raise ValueError("a bar and a verdict's set are read only with a forecaster's verdict")
    if verdict_of is not None and (not bar or verdict_of in bar or verdict_set is None):
        raise ValueError(f"{verdict_of} is read on one set against a bar of other answers")
    published_by_set: dict[str, dict[str, dict[Key, float]]] = {}
    for name, path in published:
        for answer, by_cell in read_published(path).items():
            if answer in published_by_set.setdefault(name, {}):
                raise ValueError(f"{answer} is given twice for {name}")
            published_by_set[name][answer] = by_cell
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
        verdict=verdict_of is None,
    )
    if verdict_of is not None:
        for geometric in (True, False):
            verdicts += best_bar_rows(
                sets,
                draws,
                numerator=verdict_of,
                bar=bar,
                measure="squared_mean",
                least_gain=least_gain,
                verdict_set=verdict_set if geometric else None,
                geometric=geometric,
            )
        for against in (*bar, "decoder"):
            verdicts += best_bar_rows(
                sets,
                draws,
                numerator=verdict_of,
                bar=[against],
                measure="squared_mean",
                least_gain=least_gain,
                verdict_set=None,
                geometric=True,
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
    parser.add_argument("--beside", nargs=2, action="append", default=[], metavar=("SET", "PATH"))
    parser.add_argument("--verdict-of", help="a published answer read against the bar")
    parser.add_argument("--bar", nargs="+", default=[], help="classical answers, best taken")
    parser.add_argument("--verdict-set", help="the set of months whose rows read a level")
    parser.add_argument("--least-gain", type=float, required=True)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    losses, verdicts = read(
        [(p, s, Path(path)) for p, s, path in arguments.answers],
        [(s, Path(path)) for s, path in [*arguments.chronos, *arguments.beside]],
        delta=arguments.delta,
        least_gain=arguments.least_gain,
        floor=arguments.floor,
        resamples=arguments.resamples,
        seed=arguments.seed,
        verdict_of=arguments.verdict_of,
        bar=arguments.bar,
        verdict_set=arguments.verdict_set,
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "losses.csv", LOSS_COLUMNS, losses)
    write(arguments.out / "verdicts.csv", VERDICT_COLUMNS, verdicts)
    print(markdown(LOSS_COLUMNS, losses))
    print()
    print(markdown(VERDICT_COLUMNS, verdicts))


if __name__ == "__main__":  # pragma: no cover
    main()
