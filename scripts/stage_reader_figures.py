"""Draw the stage reader's curve and power from the files the reports wrote; nothing is computed.

Two figures. The curve: what each recipe scores at every budget of labels on the units held out
of the tuning side — the mean over the cells of every campaign named for it, a band one standard
deviation wide over those cells, and each campaign's own mean as a hollow marker — one panel per
task. The power: how often the replacement rule replaces a true gain of one and two hundredths at
each budget, per kind of comparison, read directly off the draws at the seeds that ran and the
largest number of stays scored, from ``comparison_power_report.py``'s ``power.csv``.

A recipe is a candidate under its bare name together with whether it stops on held-out labels:
the stop is a regime, drawn in the recipe's hue with a dashed line, and the probe over the encoder
at its initialisation is the ink of the axes, since it is the reference everything else is read
against.

    uv run scripts/stage_reader_figures.py --scores CSV [--scores CSV ...]
        --campaign TASK LABEL CAMPAIGN_ID [--campaign ...] [--everything TASK N ...]
        [--power CSV] [--out DIR]

``--everything`` places the budget of every labelled unit on a task's axis: the count it resolved
to, which the scores do not hold.
"""

import argparse
import csv
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from scripts.campaign_pairs_report import ScoreRow, read_scores
from scripts.comparison_power_report import PowerRow, budget_order

CURVE, POWER = "stage-reader-curve.png", "stage-reader-power.png"
CAPTION = "scored on half of the tuning side; neither the validation nor the test side"
# Each recipe on its own slot of the reference categorical palette, in a fixed order whose
# adjacent pairs are documented as colour-vision safe; the stop shares its recipe's hue.
RECIPES = ("from_scratch", "full_fine_tuning", "frozen_ridge", "frozen_probe", "untrained_ridge")
COLOURS = {
    "from_scratch": "#2a78d6",
    "full_fine_tuning": "#eb6834",
    "frozen_ridge": "#1baf7a",
    "frozen_probe": "#eda100",
    "untrained_ridge": "#52514e",
}
LABELS = {
    "from_scratch": "from scratch",
    "full_fine_tuning": "full fine-tuning",
    "frozen_ridge": "probe, closed form",
    "frozen_probe": "probe, trained",
    "untrained_ridge": "probe at initialisation",
}
KIND_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GAINS_DRAWN = (0.01, 0.02)


@dataclass(frozen=True)
class Recipe:
    """A candidate under its bare name, and whether its regime stops on held-out labels."""

    base: str
    stops: bool

    @classmethod
    def of(cls, candidate: str) -> "Recipe":
        base, _, knobs = candidate.partition("@")
        turned = {knob.split("=", 1)[0] for knob in knobs.split(",") if knob}
        return cls(base, bool(turned & {"patience", "patience_steps"}))

    @property
    def label(self) -> str:
        return LABELS.get(self.base, self.base) + (", stop" if self.stops else "")

    @property
    def order(self) -> tuple[int, bool]:
        return (RECIPES.index(self.base) if self.base in RECIPES else len(RECIPES)), self.stops


@dataclass(frozen=True)
class Panel:
    """One task's cells, grouped by the campaign they came from.

    Attributes:
        task: What the panel is called.
        cells: The scores of every campaign named for the task.
        labels: Each campaign's label, by its identifier.
        everything: What the budget of every labelled unit resolved to, where it was named.
    """

    task: str
    cells: tuple[ScoreRow, ...]
    labels: Mapping[str, str]
    everything: int | None

    @property
    def by_area(self) -> bool:
        return all(c.measure == "auroc_shortfall" for c in self.cells)

    def budgets(self) -> list[str]:
        return sorted({c.budget for c in self.cells}, key=budget_order)

    def position(self, budget: str) -> int:
        """Where a budget sits on the axis of labelled units.

        Raises:
            SystemExit: If the budget of every unit is drawn and its count was not given.
        """
        if budget_order(budget) != float("inf"):
            return int(budget)
        if self.everything is None:
            raise SystemExit(
                f"{self.task} runs at every labelled unit; give --everything {self.task!r} N"
            )
        return self.everything

    def recipes(self) -> list[Recipe]:
        return sorted({Recipe.of(c.candidate) for c in self.cells}, key=lambda r: r.order)

    def line(self, recipe: Recipe) -> Iterator[tuple[int, float, float, list[float]]]:
        """At each budget the recipe has cells at: position, mean, spread, each campaign's mean."""
        for budget in self.budgets():
            cells = [
                c for c in self.cells if c.budget == budget and Recipe.of(c.candidate) == recipe
            ]
            if not cells:
                continue
            scores = [c.score for c in cells]
            per_campaign = [
                mean(c.score for c in cells if c.campaign == campaign)
                for campaign in dict.fromkeys(c.campaign for c in cells)
            ]
            spread = stdev(scores) if len(scores) > 1 else 0.0
            yield self.position(budget), mean(scores), spread, per_campaign


def panels_of(
    scores: Sequence[ScoreRow],
    campaigns: Sequence[tuple[str, str, str]],
    everything: Mapping[str, int],
) -> list[Panel]:
    """One panel per task, in the order the tasks were first named.

    Raises:
        SystemExit: If a campaign named for a task has no cell among the scores.
    """
    found = []
    for task in dict.fromkeys(task for task, _, _ in campaigns):
        labels = {campaign: label for t, label, campaign in campaigns if t == task}
        for campaign in labels:
            if not any(c.campaign == campaign for c in scores):
                raise SystemExit(f"no cell of {campaign} among the scores")
        found.append(
            Panel(
                task=task,
                cells=tuple(c for c in scores if c.campaign in labels),
                labels=labels,
                everything=everything.get(task),
            )
        )
    return found


def draw_curve(panels: Sequence[Panel], path: Path, *, caption: str = CAPTION) -> Path:
    """Draw one panel per task into ``path`` and return it.

    Raises:
        SystemExit: If there is no panel.
    """
    if not panels:
        raise SystemExit("no task to draw")
    figure, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.8), squeeze=False, constrained_layout=True
    )
    for panel, axis in zip(panels, axes[0], strict=True):
        _quiet(axis)
        for recipe in panel.recipes():
            colour = COLOURS.get(recipe.base, INK)
            style = "--" if recipe.stops else "-"
            points = list(panel.line(recipe))
            xs = [x for x, _, _, _ in points]
            ys = [y for _, y, _, _ in points]
            axis.fill_between(
                xs,
                [y - s for _, y, s, _ in points],
                [y + s for _, y, s, _ in points],
                color=colour,
                alpha=0.1,
                linewidth=0,
            )
            axis.plot(xs, ys, style, color=colour, linewidth=2, marker="o", markersize=5)
            for x, _, _, per_campaign in points:
                axis.plot(
                    [x] * len(per_campaign),
                    per_campaign,
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                    markeredgecolor=colour,
                    markeredgewidth=1,
                )
        axis.set_xscale("log")
        budgets = panel.budgets()
        axis.xaxis.set_major_locator(FixedLocator([panel.position(b) for b in budgets]))
        axis.xaxis.set_major_formatter(
            FixedFormatter(
                [
                    b if budget_order(b) != float("inf") else f"all\n({panel.position(b):,})"
                    for b in budgets
                ]
            )
        )
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("labelled units learnt from", color=INK_SOFT)
        axis.set_ylabel(
            "AUROC on the held-out units" if panel.by_area else "RMSE on the held-out units",
            color=INK_SOFT,
        )
        axis.set_title(panel.task, color=INK, fontsize=11, loc="left")
        axis.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=COLOURS.get(r.base, INK),
                    linestyle="--" if r.stops else "-",
                    linewidth=2,
                    label=r.label,
                )
                for r in panel.recipes()
            ],
            fontsize=8,
            frameon=False,
            loc="lower right" if panel.by_area else "upper right",
        )
    figure.suptitle(
        f"{caption}; a hollow marker is one pretraining's mean over seeds, "
        "the band one SD over cells",
        fontsize=8,
        color=INK_SOFT,
        x=0.01,
        ha="left",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def read_power(path: Path) -> list[PowerRow]:
    with path.open(newline="") as handle:
        return [
            PowerRow(
                kind=line["kind"],
                pair=line["pair"],
                budget=line["budget"],
                stays=int(line["stays"]),
                seeds=int(line["seeds"]),
                gain=float(line["gain"]),
                threshold=None if line["threshold"] == "" else float(line["threshold"]),
                source=line["source"],
                power=float(line["power"]),
            )
            for line in csv.DictReader(handle)
        ]


def direct_power(rows: Sequence[PowerRow]) -> dict[tuple[str, str, float], list[float]]:
    """Per kind, budget and gain, every pair's power read directly off the draws.

    At the largest number of stays scored at the budget; where a kind holds a comparison against
    several spreads at one budget, the largest spread, the one that replaces least.
    """
    direct = [r for r in rows if r.source == "direct"]
    at_budget: dict[tuple[str, str], tuple[int, float | None]] = {}
    for r in direct:
        stays, threshold = at_budget.get((r.kind, r.budget), (r.stays, r.threshold))
        if r.threshold is not None and (threshold is None or r.threshold > threshold):
            threshold = r.threshold
        at_budget[(r.kind, r.budget)] = (max(stays, r.stays), threshold)
    found: dict[tuple[str, str, float], list[float]] = {}
    for r in direct:
        if (r.stays, r.threshold) == at_budget[(r.kind, r.budget)]:
            found.setdefault((r.kind, r.budget, r.gain), []).append(r.power)
    return found


def draw_power(rows: Sequence[PowerRow], path: Path, everything: int | None = None) -> Path:
    """One panel per gain drawn: the median power over a kind's pairs at each budget.

    Raises:
        SystemExit: If no power was read directly off the draws.
    """
    found = direct_power(rows)
    if not found:
        raise SystemExit("no power read directly off the draws; nothing to draw")
    kinds = sorted({kind for kind, _, _ in found})
    budgets = sorted({budget for _, budget, _ in found}, key=budget_order)
    figure, axes = plt.subplots(
        1, len(GAINS_DRAWN), figsize=(5.0 * len(GAINS_DRAWN), 4.2), sharey=True, squeeze=False
    )
    positions = list(range(len(budgets)))
    for gain, axis in zip(GAINS_DRAWN, axes[0], strict=True):
        _quiet(axis)
        for slot, kind in enumerate(kinds):
            medians = [
                median(found[(kind, budget, gain)]) if (kind, budget, gain) in found else None
                for budget in budgets
            ]
            held = [(x, m) for x, m in zip(positions, medians, strict=True) if m is not None]
            if not held:
                continue
            axis.plot(
                [x for x, _ in held],
                [m for _, m in held],
                color=KIND_COLOURS[slot % len(KIND_COLOURS)],
                linewidth=2,
                marker="o",
                markersize=5,
            )
        axis.axhline(0.8, color=INK_SOFT, linewidth=1, linestyle=":")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                b
                if budget_order(b) != float("inf") or everything is None
                else f"all\n({everything:,})"
                for b in budgets
            ]
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("labelled units learnt from", color=INK_SOFT)
        axis.set_title(f"power at a true gain of {gain:g}", color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel("share of readings that replace", color=INK_SOFT)
    axes[0][0].legend(
        handles=[
            Line2D([], [], color=KIND_COLOURS[slot % len(KIND_COLOURS)], linewidth=2, label=kind)
            for slot, kind in enumerate(kinds)
        ],
        fontsize=8,
        frameon=False,
        loc="upper left",
        title="kind of comparison",
        title_fontsize=8,
    )
    figure.suptitle(
        "the replacement rule read directly off the draws at the seeds that ran; "
        "dotted: four times in five",
        fontsize=8,
        color=INK_SOFT,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _quiet(axis: Axes) -> None:
    """Recessive axes and grid: hairlines a step off the surface, ink only on the data."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=8)
    axis.grid(True, color=GRID, linewidth=1)
    axis.set_axisbelow(True)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, action="append", default=[], metavar="CSV")
    parser.add_argument(
        "--campaign",
        nargs=3,
        action="append",
        default=[],
        metavar=("TASK", "LABEL", "CAMPAIGN_ID"),
    )
    parser.add_argument("--everything", nargs=2, action="append", default=[], metavar=("TASK", "N"))
    parser.add_argument("--power", type=Path, default=None, metavar="CSV")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "verification" / "figures", metavar="DIR"
    )
    arguments = parser.parse_args(argv)
    if bool(arguments.scores) != bool(arguments.campaign):
        parser.error("--scores and --campaign go together")
    if not arguments.scores and arguments.power is None:
        parser.error("give scores and campaigns to draw a curve, a power file, or both")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    out: Path = arguments.out
    everything = {task: int(count) for task, count in arguments.everything}
    if arguments.scores:
        scores = [row for path in arguments.scores for row in read_scores(path)]
        drawn = draw_curve(
            panels_of(scores, [tuple(each) for each in arguments.campaign], everything),
            out / CURVE,
        )
        print(drawn)
    if arguments.power is not None:
        drawn = draw_power(
            read_power(arguments.power), out / POWER, next(iter(everything.values()), None)
        )
        print(drawn)


if __name__ == "__main__":
    main()
