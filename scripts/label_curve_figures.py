"""Draw scores over labelled units, and gains with their intervals, from the pairs report's files.

Two panels over one log axis of labelled units. The upper one draws each named line: a candidate
under one or more campaigns, its score per seed the mean over those campaigns, the line the mean
over seeds and the band ±1.96 standard errors over seeds. A line held at a single budget is drawn
as a point. The lower one draws each named gain as the pairs report stated it, mean and interval,
beside zero and the least gain the claim was judged by. Nothing is trained or resampled here: the
scores come from ``scores.csv`` and the gains from ``comparisons.csv``.

    uv run scripts/label_curve_figures.py --figure PATH
        --scores FILE [--scores FILE ...] --comparisons FILE [--comparisons FILE ...]
        --line LABEL CAMPAIGNS CANDIDATE [--line ...]
        --gain LABEL CONTROL_CAMPAIGNS CONTROL CANDIDATE_CAMPAIGNS CANDIDATE [--gain ...]
        [--least-gain 0.02] [--title TEXT] [--units stays]

Campaigns of one side are joined by ``+``. The table of what was drawn is printed for the note.
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
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

from scripts.campaign_pairs_report import GROUPED_BY, ScoreRow, read_scores
from scripts.pretraining_screen_report import Pair, read_pairs

# The reference categorical palette in its documented order, whose adjacent pairs are
# colour-vision safe (the order the campaign curve uses); the sixth hue lies outside that set.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7")
# The areas this figure is labelled in: the pairs report's measure for a ranking.
MEASURE = "auroc_shortfall"
INK, INK_SOFT = "#0b0b0b", "#52514e"
# Gains are drawn in ink, told apart by marker, so no colour in the lower panel repeats a line's.
GAIN_STYLES = ((INK, "o"), ("#8a8986", "s"), (INK_SOFT, "^"), ("#b3b2ae", "D"))
Z = 1.96


@dataclass(frozen=True, kw_only=True)
class Point:
    """A line's score at one budget: the mean over seeds and its standard error."""

    budget: int
    mean: float
    se: float
    seeds: int


@dataclass(frozen=True, kw_only=True)
class Line:
    """A named candidate under one or more campaigns, read at every budget it was scored at."""

    label: str
    campaigns: str
    candidate: str

    def points(self, scores: Sequence[ScoreRow]) -> list[Point]:
        """The line's points, budget by budget.

        A budget may live in some of the campaigns only (a classical bar read in two runs), but
        every campaign must hold the candidate somewhere, and the campaigns at a budget must hold
        the same seeds, so no seed's mean is taken over fewer pretrainings than another's.

        Raises:
            ValueError: If no score belongs to it, a campaign holds none, the campaigns at a
                budget hold different seeds, a budget is not a count of units, or a score is not
                an area.
        """
        campaigns = set(self.campaigns.split(GROUPED_BY))
        by_budget: dict[int, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for row in scores:
            if row.campaign in campaigns and row.candidate == self.candidate:
                if row.measure != MEASURE:
                    raise ValueError(f"{self.label} is scored by {row.measure}, not an area")
                if not row.budget.isdigit():
                    raise ValueError(f"budget {row.budget!r} of {self.label} is not a count")
                by_budget[int(row.budget)][row.campaign][row.seed] = row.score
        if not by_budget:
            raise ValueError(f"no score of {self.candidate} under {self.campaigns}")
        silent = campaigns - {c for held in by_budget.values() for c in held}
        if silent:
            raise ValueError(f"no score of {self.label} under {sorted(silent)[0]}")
        points = []
        for budget in sorted(by_budget):
            seeds = {frozenset(held) for held in by_budget[budget].values()}
            if len(seeds) > 1:
                raise ValueError(f"the campaigns of {self.label} hold different seeds at {budget}")
            per_seed = [
                mean(held[seed] for held in by_budget[budget].values())
                for seed in next(iter(seeds))
            ]
            se = stdev(per_seed) / sqrt(len(per_seed)) if len(per_seed) > 1 else 0.0
            points.append(Point(budget=budget, mean=mean(per_seed), se=se, seeds=len(per_seed)))
        return points


@dataclass(frozen=True, kw_only=True)
class Gain:
    """A named comparison the pairs report made, at every budget it was made at."""

    label: str
    control_campaigns: str
    control: str
    candidate_campaigns: str
    candidate: str

    def pairs(self, pairs: Sequence[Pair]) -> list[Pair]:
        """The comparison's rows, by budget.

        Raises:
            ValueError: If the report holds none of it.
        """
        found = [
            p
            for p in pairs
            if (p.control_campaign, p.control, p.candidate_campaign, p.candidate)
            == (self.control_campaigns, self.control, self.candidate_campaigns, self.candidate)
        ]
        if not found:
            raise ValueError(f"no comparison {self.label!r} among the comparisons")
        return sorted(found, key=lambda p: int(p.budget))


def draw(
    lines: Sequence[tuple[Line, list[Point]]],
    gains: Sequence[tuple[Gain, list[Pair]]],
    path: Path,
    *,
    least_gain: float,
    title: str,
    units: str = "units",
) -> Path:
    """Draw both panels into ``path`` and return it."""
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(7.5, 8.0), sharex=True, height_ratios=(3, 2)
    )
    budgets = sorted(
        {p.budget for _, points in lines for p in points}
        | {int(p.budget) for _, pairs in gains for p in pairs}
    )
    for index, (line, points) in enumerate(lines):
        colour = PALETTE[index % len(PALETTE)]
        xs = [p.budget for p in points]
        means = [p.mean for p in points]
        if len(points) == 1:
            upper.errorbar(
                xs,
                means,
                yerr=[Z * points[0].se],
                color=colour,
                fmt="D",
                markersize=6,
                capsize=3,
                label=line.label,
            )
            continue
        upper.plot(xs, means, color=colour, linewidth=2, marker="o", markersize=5, label=line.label)
        upper.fill_between(
            xs,
            [p.mean - Z * p.se for p in points],
            [p.mean + Z * p.se for p in points],
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
    upper.set_ylabel("area under the ROC curve")
    upper.legend(
        fontsize=8,
        loc="upper left",
        title="mean over seeds, band = mean ± 1.96 SE over seeds",
        title_fontsize=8,
    )
    upper.grid(True, alpha=0.25)

    offsets = [1.0 + 0.07 * (i - (len(gains) - 1) / 2) for i in range(len(gains))]
    for index, ((gain, pairs), offset) in enumerate(zip(gains, offsets, strict=True)):
        shifted = [int(p.budget) * offset for p in pairs]
        colour, marker = GAIN_STYLES[index % len(GAIN_STYLES)]
        lower.errorbar(
            shifted,
            [p.mean for p in pairs],
            yerr=[[p.mean - p.low for p in pairs], [p.high - p.mean for p in pairs]],
            color=colour,
            fmt=marker,
            markersize=5,
            capsize=3,
            linewidth=1.5,
            label=gain.label,
        )
    lower.axhline(0.0, color=INK_SOFT, linewidth=1)
    lower.axhline(least_gain, color=INK, linewidth=1, linestyle="--", label="least gain")
    bottom, top = lower.get_ylim()
    lower.set_ylim(bottom, top + 0.5 * (top - bottom))
    lower.legend(fontsize=8, loc="upper left", frameon=False)
    lower.set_ylabel("gain in area, 95 % interval")
    lower.set_xscale("log")
    lower.set_xticks(budgets)
    lower.set_xticklabels([str(b) for b in budgets], fontsize=8)
    lower.minorticks_off()
    lower.set_xlabel(f"labelled {units} in the budget")
    lower.grid(True, alpha=0.25)
    upper.margins(x=0.12)

    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def render(
    lines: Sequence[tuple[Line, list[Point]]], gains: Sequence[tuple[Gain, list[Pair]]]
) -> str:
    """The drawn numbers as the tables a note pastes."""
    out = ["| Line | Budget | Mean | SE | Seeds |", "| --- | --- | --- | --- | --- |"]
    for line, points in lines:
        for point in points:
            out.append(
                f"| {line.label} | {point.budget} | {point.mean:.4f} | {point.se:.4f} "
                f"| {point.seeds} |"
            )
    out += ["", "| Gain | Budget | Mean | Interval |", "| --- | --- | --- | --- |"]
    for gain, pairs in gains:
        for pair in pairs:
            out.append(
                f"| {gain.label} | {pair.budget} | {pair.mean:+.4f} "
                f"| [{pair.low:+.4f}; {pair.high:+.4f}] |"
            )
    return "\n".join(out)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--comparisons", type=Path, action="append", default=[])
    parser.add_argument(
        "--line",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CAMPAIGNS", "CANDIDATE"),
    )
    parser.add_argument(
        "--gain",
        nargs=5,
        action="append",
        default=[],
        metavar=("LABEL", "CONTROL_CAMPAIGNS", "CONTROL", "CANDIDATE_CAMPAIGNS", "CANDIDATE"),
    )
    parser.add_argument("--least-gain", type=float, default=0.02)
    parser.add_argument("--title", default="Label efficiency — validation, not test")
    parser.add_argument("--units", default="units", help="what the budget counts, for the axis")
    arguments = parser.parse_args(argv)
    if arguments.gain and not arguments.comparisons:
        parser.error("a --gain needs at least one --comparisons file")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    scores = [row for path in arguments.scores for row in read_scores(path)]
    pairs = [p for path in arguments.comparisons for p in read_pairs(path)]
    lines = [
        (line, line.points(scores))
        for line in (Line(label=a, campaigns=b, candidate=c) for a, b, c in arguments.line)
    ]
    gains = [
        (gain, gain.pairs(pairs))
        for gain in (
            Gain(label=a, control_campaigns=b, control=c, candidate_campaigns=d, candidate=e)
            for a, b, c, d, e in arguments.gain
        )
    ]
    draw(
        lines,
        gains,
        arguments.figure,
        least_gain=arguments.least_gain,
        title=arguments.title,
        units=arguments.units,
    )
    print(render(lines, gains))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
