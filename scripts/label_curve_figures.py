"""Draw the label-efficiency curve from the files the curve report stored, never from the runs.

Two panels over one axis of budgets. The upper one is the curve itself: the validation error of
every transfer mode at every budget, the mean over seeds with the spread between seeds as a
band, and the mean predictor as the line any candidate has to get under. The lower one is what
the registered rules judge: the reduction of the error against the control arm, with the paired
interval over units, the endpoint ringed, and the practical floor of each budget as a band
around zero, so that a reduction the floor swallows reads as nil where it is drawn. The budgets
are labelled with how many units the labels came from, because windows of one unit overlap and
the count of windows overstates what a budget holds. The task names the words: what its units
are called, what its error is measured in, and what the figure is titled.

    uv run scripts/label_curve_figures.py data/report/curve/<name>
    uv run scripts/label_curve_figures.py <dir> --figure docs/verification/figures/<name>.png
    uv run scripts/label_curve_figures.py <dir> --task null-b-forecast
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import NamedTuple

import matplotlib

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A file is written, nothing is shown: the backend must not need a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from scripts.label_curve_report import Curve, read
from scripts.transfer_grid import MODES, KnownTask, KnownTasks, budget_rank

STEM = "label-efficiency-curve"
CAPTION = "tier M, Kaggle T4, fp32 — preliminary; validation, not test"
# One hue per mode in a fixed order, so a mode keeps its colour whichever modes a figure shows.
COLOURS = dict(zip(MODES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True))
LABELS = {
    "from_scratch": "from scratch",
    "frozen_probe": "frozen probe",
    "lora": "LoRA",
    "full_fine_tuning": "full fine-tuning",
}
# How a task is named in a title; a task not listed here is titled by its name.
TITLES = {
    "turbofan-fd001": "turbofan FD001",
    "control-b-forecast": "the coupled synthetic pair",
    "null-b-forecast": "the null synthetic pair",
    "control-b-wide-forecast": "the coupled synthetic pair",
    "null-b-wide-forecast": "the null synthetic pair",
    "control-b-shared-forecast": "the synthetic pair over shared trajectories (a ceiling)",
}


class Wording(NamedTuple):
    """The words a task puts on the figure."""

    title: str
    units: str
    error_unit: str


def wording_of(task: KnownTask) -> Wording:
    """What the figure calls the task, its units and the unit of its error.

    A remaining-life target is counted in cycles; a forecast is in the units of the sensor it
    forecasts, which the synthetic corpora leave nameless.
    """
    error_unit = "cycles" if isinstance(task.labels, RemainingLifeScheme) else "sensor units"
    return Wording(TITLES.get(task.name, task.name), task.units_called, error_unit)


def draw(
    curve: Curve, path: Path, *, task: KnownTask | None = None, caption: str = CAPTION
) -> Path:
    """Draw the curve and the comparisons into ``path``, worded for ``task``."""
    words = wording_of(KnownTasks.default() if task is None else task)
    budgets = sorted({point.budget for point in curve.points}, key=budget_rank)
    windows = {budget: _windows(curve, budget) for budget in budgets}
    engines = {
        budget: sorted({p.engines for p in curve.points if p.budget == budget})
        for budget in budgets
    }
    positions = [windows[budget] for budget in budgets]

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True, height_ratios=(3, 2)
    )
    ends: list[tuple[str, float, float]] = []
    for mode in MODES:
        # A partial grid leaves a mode without some budgets; its line runs over those it has.
        rows = {
            budget: [p.rmse for p in curve.points if p.mode == mode and p.budget == budget]
            for budget in budgets
        }
        held = [budget for budget in budgets if rows[budget]]
        if not held:
            continue
        xs = [windows[budget] for budget in held]
        means = [mean(rows[budget]) for budget in held]
        spread = [stdev(rows[budget]) if len(rows[budget]) > 1 else 0.0 for budget in held]
        upper.plot(
            xs,
            means,
            color=COLOURS[mode],
            linewidth=2,
            marker="o",
            markersize=5,
            label=LABELS[mode],
        )
        upper.fill_between(
            xs,
            [m - s for m, s in zip(means, spread, strict=True)],
            [m + s for m, s in zip(means, spread, strict=True)],
            color=COLOURS[mode],
            alpha=0.15,
            linewidth=0,
        )
        ends.append((mode, xs[-1], means[-1]))
    for baseline in curve.baselines:
        if baseline.name == "mean predictor":
            upper.axhline(baseline.rmse, color="#52514e", linestyle=":", linewidth=1)
            upper.annotate(
                "mean predictor",
                (positions[len(positions) // 2], baseline.rmse),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
                color="#52514e",
                ha="center",
            )
    upper.set_ylabel(f"validation RMSE ({words.error_unit})")
    upper.set_ylim(bottom=0)
    upper.margins(x=0.12)
    bottom, top = upper.get_ylim()
    # A line of the labels' text is about a thirtieth of the panel's height.
    heights = label_heights([end for _, _, end in ends], (top - bottom) / 30)
    for (mode, x, _), height in zip(ends, heights, strict=True):
        upper.annotate(
            LABELS[mode],
            (x, height),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color="#52514e",
            va="center",
            annotation_clip=False,
        )
    upper.grid(True, alpha=0.25)
    upper.legend(
        fontsize=8, loc="lower left", title="mean over seeds; band = ±1 SD", title_fontsize=8
    )

    offsets = {"frozen_probe": 0.92, "lora": 1.0, "full_fine_tuning": 1.08}
    for mode in MODES:
        compared = [row for row in curve.comparisons if row.mode == mode]
        if not compared:
            continue
        xs = [windows[row.budget] * offsets.get(mode, 1.0) for row in compared]
        lower.errorbar(
            xs,
            [row.reduction for row in compared],
            yerr=[
                [row.reduction - row.low for row in compared],
                [row.high - row.reduction for row in compared],
            ],
            color=COLOURS[mode],
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.5,
            label=LABELS[mode],
        )
        for x, row in zip(xs, compared, strict=True):
            if row.primary:
                lower.plot(
                    x, row.reduction, marker="o", markersize=12, mfc="none", mec="#0b0b0b", mew=1.5
                )
                lower.annotate(
                    "registered endpoint",
                    (x, row.low),
                    xytext=(0, -10),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                    va="top",
                    color="#0b0b0b",
                )
    for budget in budgets:
        floors = [row.floor for row in curve.comparisons if row.budget == budget]
        if floors:
            x = windows[budget]
            lower.fill_between(
                [x * 0.86, x * 1.14],
                -max(floors),
                max(floors),
                color="#52514e",
                alpha=0.12,
                linewidth=0,
                label="practical floor" if budget == budgets[0] else None,
            )
    lower.axhline(0.0, color="#52514e", linewidth=1)
    lower.legend(fontsize=8, loc="lower left")
    lower.set_ylabel(f"RMSE reduction vs from scratch\n(95 % interval over {words.units})")
    lower.set_xscale("log")
    lower.set_xticks(positions)
    lower.set_xticklabels(
        [f"{budget}\n({_range(engines[budget])} {words.units})" for budget in budgets], fontsize=8
    )
    lower.minorticks_off()
    lower.set_xlabel("labelled windows in the budget")
    lower.grid(True, alpha=0.25)

    figure.suptitle(f"Label efficiency on {words.title} — {caption}", fontsize=10)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _windows(curve: Curve, budget: str) -> float:
    """Where a budget sits on the axis: the count of windows it resolved to."""
    return float(max(p.windows for p in curve.points if p.budget == budget))


def label_heights(ends: Sequence[float], gap: float) -> list[float]:
    """Where each line's label sits: at the line's end, or raised clear of the label below it.

    Lines of a curve often end within a few tenths of each other, and labels written at their
    ends would print over one another.
    """
    heights = list(ends)
    below = -math.inf
    for index in sorted(range(len(ends)), key=lambda each: ends[each]):
        heights[index] = max(ends[index], below + gap)
        below = heights[index]
    return heights


def _range(values: Sequence[int]) -> str:
    return f"{values[0]}-{values[-1]}" if len(values) > 1 else str(values[0])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curve", type=Path, help="directory the report stored the curve under")
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help=f"file to draw into; {STEM}.png beside the curve unless given",
    )
    parser.add_argument(
        "--task",
        choices=KnownTasks.names(),
        default=KnownTasks.default().name,
        help="the task the curve was run on, which words the figure",
    )
    parser.add_argument("--caption", default=CAPTION, help="tier, platform and precision")
    arguments = parser.parse_args(argv)
    figure = arguments.figure or arguments.curve / f"{STEM}.png"
    print(
        draw(
            read(arguments.curve),
            figure,
            task=KnownTasks.named(arguments.task),
            caption=arguments.caption,
        )
    )


if __name__ == "__main__":
    main()
