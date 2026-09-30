"""How far a cell's area moves between its seeds, beside how far the units it scored move it.

A floor read in area has to sit above the noise a seed makes, and a campaign's repeats carry
two kinds of it: what the seed changes in the model, and, where each repeat scores other units,
which units were scored. This script reads the answers ``campaign_pairs_report.py`` exported and
gives every candidate at every budget of every campaign its area under each seed, the standard
deviation of those areas, and the standard error of one area over its own units — a bootstrap in
two strata, units holding a positive and units that do not, as the verdict resamples them. Per
budget it pools the sides: the typical spread, the typical standard error, and the spread left
once each seed's shift shared by every side is taken out, which is what remains of a seed where
every side of that seed scored the same units.

    uv run scripts/campaign_pairs_report.py --campaign ID [--campaign ID ...] --out DIR
    uv run scripts/seed_spread_report.py --out DIR [--resamples N]

Writes ``spreads.csv`` beside the exported answers and prints the tables a note pastes.
"""

import argparse
import csv
import random
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from scripts.campaign_pairs_report import PREDICTIONS, Answers, Side, read_predictions
from scripts.reporting import table

SPREADS = "spreads.csv"
BUDGET_ORDER = ("50", "200", "1000", "all")


@dataclass(frozen=True, kw_only=True)
class Spread:
    """One side's area under each of its seeds, and one area's standard error over its units."""

    side: Side
    seeds: tuple[int, ...]
    areas: tuple[float, ...]
    unit_error: float

    @property
    def mean(self) -> float:
        return statistics.fmean(self.areas)

    @property
    def deviation(self) -> float:
        return statistics.stdev(self.areas)


@dataclass(frozen=True, kw_only=True)
class Pooled:
    """Every side of one budget read together."""

    budget: str
    sides: int
    median_deviation: float
    typical_deviation: float
    typical_unit_error: float
    residual: float | None


def unit_error(ranking: WindowRanking, resamples: int, draws: random.Random) -> float:
    """The standard deviation of the area over resamples of the ranked units, in two strata."""
    holding = {ranking.sequence[place] for place, hit in enumerate(ranking.positive) if hit}
    strata = [
        [unit for unit in range(len(ranking.units)) if unit in holding],
        [unit for unit in range(len(ranking.units)) if unit not in holding],
    ]
    areas = []
    for _ in range(resamples):
        weights = [0] * len(ranking.units)
        for stratum in strata:
            for pick in draws.choices(stratum, k=len(stratum)):
                weights[pick] += 1
        areas.append(ranking.auroc_weighted(weights))
    return statistics.stdev(areas)


def spreads(answers: Answers, resamples: int, seed: int) -> list[Spread]:
    """Every exported side with at least two seeds, read seed by seed."""
    draws = random.Random(seed)
    found: list[Spread] = []
    for side, seeds in sorted(answers.sides().items()):
        if len(seeds) < 2:
            continue
        rankings = [answers.ranking(side, each) for each in seeds]
        found.append(
            Spread(
                side=side,
                seeds=tuple(seeds),
                areas=tuple(ranking.auroc for ranking in rankings),
                unit_error=sqrt(
                    statistics.fmean(
                        unit_error(ranking, resamples, draws) ** 2 for ranking in rankings
                    )
                ),
            )
        )
    return found


def pooled(found: Sequence[Spread]) -> list[Pooled]:
    """Per budget, the sides' spreads and standard errors read together.

    The residual takes each seed's shift out, which only means something where every side of
    the budget ran the same seeds; elsewhere it is left unread.
    """
    rows = []
    budgets = sorted({s.side.budget for s in found}, key=_budget_order)
    for budget in budgets:
        sides = [s for s in found if s.side.budget == budget]
        deviations = [s.deviation for s in sides]
        rows.append(
            Pooled(
                budget=budget,
                sides=len(sides),
                median_deviation=statistics.median(deviations),
                typical_deviation=sqrt(statistics.fmean(d * d for d in deviations)),
                typical_unit_error=sqrt(statistics.fmean(s.unit_error**2 for s in sides)),
                residual=_residual(sides),
            )
        )
    return rows


def _residual(sides: Sequence[Spread]) -> float | None:
    """The spread left once each seed's shift, common to every side, is removed."""
    if len(sides) < 2 or len({s.seeds for s in sides}) != 1:
        return None
    count = len(sides[0].seeds)
    shift = [statistics.fmean(s.areas[k] - s.mean for s in sides) for k in range(count)]
    squares = sum((s.areas[k] - s.mean - shift[k]) ** 2 for s in sides for k in range(count))
    return sqrt(squares / ((len(sides) - 1) * (count - 1)))


def _budget_order(budget: str) -> tuple[int, str]:
    return (
        (BUDGET_ORDER.index(budget), "") if budget in BUDGET_ORDER else (len(BUDGET_ORDER), budget)
    )


def write(path: Path, found: Sequence[Spread]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("campaign", "candidate", "budget", "seeds", "areas", "mean", "sd", "unit_error")
        )
        for s in found:
            writer.writerow(
                (
                    s.side.campaign,
                    s.side.candidate,
                    s.side.budget,
                    " ".join(map(str, s.seeds)),
                    " ".join(f"{a:.6f}" for a in s.areas),
                    f"{s.mean:.6f}",
                    f"{s.deviation:.6f}",
                    f"{s.unit_error:.6f}",
                )
            )


def render(found: Sequence[Spread]) -> str:
    sides = table(
        ("campaign", "candidate", "budget", "seeds", "mean area", "SD over seeds", "SE over units"),
        (
            (
                s.side.campaign[:8],
                s.side.candidate,
                s.side.budget,
                str(len(s.seeds)),
                f"{s.mean:.3f}",
                f"{s.deviation:.4f}",
                f"{s.unit_error:.4f}",
            )
            for s in sorted(found, key=lambda s: (s.side.campaign, _budget_order(s.side.budget)))
        ),
    )
    budgets = table(
        (
            "budget",
            "sides",
            "median SD",
            "RMS SD",
            "RMS SE over units",
            "SD without the seed's shift",
        ),
        (
            (
                p.budget,
                str(p.sides),
                f"{p.median_deviation:.4f}",
                f"{p.typical_deviation:.4f}",
                f"{p.typical_unit_error:.4f}",
                "—" if p.residual is None else f"{p.residual:.4f}",
            )
            for p in pooled(found)
        ),
    )
    return f"{sides}\n\n{budgets}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--resamples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1, help="seed of the bootstrap's draws")
    arguments = parser.parse_args(argv)
    source = arguments.out / PREDICTIONS
    if not source.exists():
        raise SystemExit(f"no exported answers in {source}: run campaign_pairs_report.py first")
    found = spreads(Answers(read_predictions(source)), arguments.resamples, arguments.seed)
    if not found:
        raise SystemExit("no side was run under two seeds or more")
    write(arguments.out / SPREADS, found)
    print(render(found))


if __name__ == "__main__":
    main()
