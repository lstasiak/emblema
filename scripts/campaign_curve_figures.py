"""Draw a campaign's curve from the files the campaign report wrote; nothing is computed here.

Three panels over one axis of labelled windows: the networks' error, the classical baselines'
error beside the control, and the pretrained arms' reduction against the control with the
interval and the practical floor the campaign judged them by. Each candidate keeps its hue
whichever candidates a campaign holds, so two figures read alike.

    uv run scripts/campaign_curve_figures.py data/report/<dir> [--figure PATH] [--caption TEXT]
"""

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from scripts.campaign_report import CELLS, COMPARISONS, CellRow, ComparisonRow

STEM = "label-efficiency-curve"
CAPTION = "tier M — preliminary; validation, not test"
# The five arms in reporting order, each on its own slot of the reference categorical palette:
# the first five of a fixed order whose adjacent pairs are documented as colour-vision safe.
ARMS = ("from_scratch", "frozen_probe", "frozen_ridge", "lora", "full_fine_tuning")
COLOURS = dict(zip(ARMS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"), strict=True))
LABELS = {
    "from_scratch": "from scratch",
    "frozen_probe": "frozen probe",
    "frozen_ridge": "frozen probe, closed form",
    "lora": "LoRA",
    "full_fine_tuning": "full fine-tuning",
    "boosted_trees_per_channel": "trees per channel",
    "boosted_trees_spectral": "trees over the spectrum",
    "boosted_trees_across_channels": "trees across channels",
    "minirocket": "MiniRocket",
    "patch_transformer": "patch model",
}
INK, INK_SOFT = "#0b0b0b", "#52514e"
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


@dataclass(frozen=True)
class Report:
    """The two files of a campaign report, read."""

    cells: tuple[CellRow, ...]
    comparisons: tuple[ComparisonRow, ...]


def base_of(candidate: str) -> str:
    """The candidate under its bare name, whatever knobs its variant turns."""
    return candidate.split("@", 1)[0]


def read(directory: Path) -> Report:
    """The report stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CELLS).is_file():
        raise SystemExit(f"{directory} holds no {CELLS}; nothing to draw")
    with (directory / CELLS).open(newline="") as handle:
        if "rmse" not in (csv.DictReader(handle).fieldnames or ()):
            raise SystemExit(
                f"{directory} reports a campaign read by area under the ROC curve; this draws "
                "the curve of an error in the task's unit"
            )
    with (directory / CELLS).open(newline="") as handle:
        cells = tuple(
            CellRow(
                candidate=row["candidate"],
                kind=row["kind"],
                budget=row["budget"],
                windows=int(row["windows"]),
                seed=int(row["seed"]),
                units=int(row["units"]),
                score=float(row["rmse"]),
                brier=None,
                seconds=float(row["seconds"]),
            )
            for row in csv.DictReader(handle)
        )
    with (directory / COMPARISONS).open(newline="") as handle:
        comparisons = tuple(
            ComparisonRow(
                candidate=row["candidate"],
                budget=row["budget"],
                windows=int(row["windows"]),
                repeats=int(row["repeats"]),
                control_score=float(row["control_rmse"]),
                control_sd=float(row["control_sd"]),
                candidate_score=float(row["candidate_rmse"]),
                candidate_sd=float(row["candidate_sd"]),
                reduction=float(row["reduction"]),
                relative_reduction=(
                    None if row["relative_reduction"] == "" else float(row["relative_reduction"])
                ),
                low=float(row["low"]),
                high=float(row["high"]),
                p_value=float(row["p_value"]),
                floor=float(row["floor"]),
                primary=row["primary"],
                verdict=row["verdict"],
            )
            for row in csv.DictReader(handle)
        )
    return Report(cells=cells, comparisons=comparisons)


def draw(report: Report, path: Path, *, caption: str = CAPTION) -> Path:
    """Draw the panels into ``path`` and return it: the baselines' panel only where there are any.

    Raises:
        SystemExit: If the report holds no cell.
    """
    if not report.cells:
        raise SystemExit("the report holds no cell; nothing to draw")
    budgets = sorted({c.budget for c in report.cells}, key=lambda b: _windows(report, b))
    positions = [_windows(report, budget) for budget in budgets]
    candidates = list(dict.fromkeys(c.candidate for c in report.cells))
    arms = [c for c in candidates if base_of(c) in ARMS]
    arms.sort(key=lambda c: ARMS.index(base_of(c)))
    baselines = [c for c in candidates if base_of(c) not in ARMS]
    control = next((c for c in arms if base_of(c) == "from_scratch"), None)

    if baselines:
        figure, (upper, middle, lower) = plt.subplots(
            3, 1, figsize=(7.5, 10.5), sharex=True, height_ratios=(3, 2.4, 2)
        )
    else:
        figure, (upper, lower) = plt.subplots(
            2, 1, figsize=(7.5, 7.5), sharex=True, height_ratios=(3, 2)
        )
        middle = None
    ends = []
    for candidate in arms:
        line = _line(report, candidate, budgets)
        if not line:
            continue
        xs, means, spread = line
        colour = COLOURS[base_of(candidate)]
        upper.plot(xs, means, color=colour, linewidth=2, marker="o", markersize=5)
        upper.fill_between(
            xs,
            [m - s for m, s in zip(means, spread, strict=True)],
            [m + s for m, s in zip(means, spread, strict=True)],
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
        ends.append((LABELS[base_of(candidate)], xs[-1], means[-1]))
    _label_ends(upper, ends)
    upper.set_ylabel("validation RMSE (cycles)")
    upper.legend(
        handles=[
            Line2D([], [], color=COLOURS[base_of(c)], linewidth=2, label=LABELS[base_of(c)])
            for c in arms
        ],
        fontsize=8,
        loc="lower left",
        title="networks; mean over seeds, band = ±1 SD",
        title_fontsize=8,
    )
    upper.grid(True, alpha=0.25)

    if middle is not None:
        _draw_baselines(middle, report, budgets, baselines, control)

    compared_arms = [c for c in arms if base_of(c) != "from_scratch"]
    offsets = {c: 0.88 + 0.08 * i for i, c in enumerate(compared_arms)}
    for candidate in compared_arms:
        rows = [r for r in report.comparisons if r.candidate == candidate]
        if not rows:
            continue
        rows.sort(key=lambda r: r.windows)
        shifted = [r.windows * offsets[candidate] for r in rows]
        lower.errorbar(
            shifted,
            [r.reduction for r in rows],
            yerr=[[r.reduction - r.low for r in rows], [r.high - r.reduction for r in rows]],
            color=COLOURS[base_of(candidate)],
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.5,
            label=LABELS[base_of(candidate)],
        )
        for x, row in zip(shifted, rows, strict=True):
            if row.primary == "yes":
                lower.plot(
                    x, row.reduction, marker="o", markersize=12, mfc="none", mec=INK, mew=1.5
                )
                lower.annotate(
                    "registered endpoint",
                    (x, row.low),
                    xytext=(0, -10),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                    va="top",
                    color=INK,
                )
    for budget, x in zip(budgets, positions, strict=True):
        floors = [r.floor for r in report.comparisons if r.budget == budget]
        if floors:
            lower.fill_between(
                [x * 0.84, x * 1.16],
                -max(floors),
                max(floors),
                color=INK_SOFT,
                alpha=0.12,
                linewidth=0,
                label="practical floor" if budget == budgets[0] else None,
            )
    lower.axhline(0.0, color=INK_SOFT, linewidth=1)
    if lower.get_legend_handles_labels()[0]:
        # The legend sits in a band above the data, so it covers no interval.
        bottom, top = lower.get_ylim()
        lower.set_ylim(bottom, top + 0.45 * (top - bottom))
        lower.legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
    lower.set_ylabel("RMSE reduction vs from scratch\n(95 % interval over engines)")
    lower.set_xscale("log")
    lower.set_xticks(positions)
    lower.set_xticklabels(
        [f"{b}\n({_windows(report, b):,})" if b == "all" else b for b in budgets], fontsize=8
    )
    lower.minorticks_off()
    lower.set_xlabel("labelled windows in the budget")
    lower.grid(True, alpha=0.25)
    for panel in (upper, middle) if middle is not None else (upper,):
        panel.set_ylim(bottom=0)
        panel.margins(x=0.12)

    figure.suptitle(f"Label efficiency on the turbofan task — {caption}", fontsize=10)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _draw_baselines(
    middle: Axes,
    report: Report,
    budgets: Sequence[str],
    baselines: Sequence[str],
    control: str | None,
) -> None:
    """The classical baselines in ink, told apart by marker and name, the control dotted beside."""
    ends = []
    if control is not None and (line := _line(report, control, budgets)):
        middle.plot(line[0], line[1], color=INK_SOFT, linestyle=":", linewidth=1.5)
        ends.append((LABELS["from_scratch"], line[0][-1], line[1][-1]))
    for index, candidate in enumerate(baselines):
        line = _line(report, candidate, budgets)
        if not line:
            continue
        xs, means, _ = line
        middle.plot(
            xs,
            means,
            color=INK,
            linewidth=1.2,
            marker=MARKERS[index % len(MARKERS)],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        ends.append((LABELS.get(base_of(candidate), base_of(candidate)), xs[-1], means[-1]))
    _label_ends(middle, ends)
    middle.set_ylabel("validation RMSE (cycles)")
    middle.legend(
        handles=[
            Line2D(
                [],
                [],
                color=INK,
                marker=MARKERS[i % len(MARKERS)],
                markerfacecolor="white",
                linewidth=1.2,
                label=LABELS.get(base_of(c), base_of(c)),
            )
            for i, c in enumerate(baselines)
        ]
        + (
            [Line2D([], [], color=INK_SOFT, linestyle=":", label="from scratch")] if control else []
        ),
        fontsize=8,
        loc="lower left",
        title="classical baselines beside the control; mean over seeds",
        title_fontsize=8,
    )
    middle.grid(True, alpha=0.25)


def _windows(report: Report, budget: str) -> int:
    return next(c.windows for c in report.cells if c.budget == budget)


def _line(
    report: Report, candidate: str, budgets: Sequence[str]
) -> tuple[list[int], list[float], list[float]] | None:
    """The mean and spread of a candidate's error at every budget it has cells at."""
    rows = {
        budget: [c.score for c in report.cells if c.candidate == candidate and c.budget == budget]
        for budget in budgets
    }
    held = [budget for budget in budgets if rows[budget]]
    if not held:
        return None
    return (
        [_windows(report, budget) for budget in held],
        [mean(rows[budget]) for budget in held],
        [stdev(rows[budget]) if len(rows[budget]) > 1 else 0.0 for budget in held],
    )


def _label_ends(panel: Axes, ends: Sequence[tuple[str, float, float]]) -> None:
    """Name each line where it ends, the names set apart where the ends crowd."""
    if not ends:
        return
    panel.set_ylim(bottom=0)
    bottom, top = panel.get_ylim()
    heights = label_heights([end for _, _, end in ends], (top - bottom) / 24)
    for (label, x, _), height in zip(ends, heights, strict=True):
        panel.annotate(
            label,
            (x, height),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK_SOFT,
            va="center",
            annotation_clip=False,
        )


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="directory the campaign report wrote")
    parser.add_argument(
        "--figure", type=Path, default=None, help=f"file to draw into; {STEM}.png beside the report"
    )
    parser.add_argument("--caption", default=CAPTION, help="tier, platform and precision")
    arguments = parser.parse_args(argv)
    figure = arguments.figure or arguments.report / f"{STEM}.png"
    print(draw(read(arguments.report), figure, caption=arguments.caption))


if __name__ == "__main__":
    main()
