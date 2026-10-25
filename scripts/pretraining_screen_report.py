"""The declared reading of a screen of pretraining recipes, over the pairs already compared.

Each recipe is pretrained a few times and every pretraining fine-tuned in its own campaign beside
the network from nothing. ``campaign_pairs_report.py`` pairs the sides — a side of several
campaigns joined by ``+`` is the mean over those pretrainings — and writes ``comparisons.csv``;
this script reads those rows by the rule written down before the run. The rules themselves —
the spread, the thresholds, the levels and when a contrast moves a recipe — are the Evaluation
context's; what is here is which rows they read and the shape of a note.

- *Spread.* s is the larger of a floor and fine-tuning's own spread between pretrainings of one
  recipe, by the registered formula: over the pairs given at every budget read, the squared mean
  difference less its noise (the squared standard error over seeds plus the square of the paired
  interval's half-width over 1.96), averaged over pairs and budgets, floored at zero, then its
  root.
- *Gains.* Fine-tuning less the network from nothing under n pretrainings reads against
  T = 2·√(SE² + (s²/2)/n), the side from nothing carrying no spread of pretraining. Its level:
  at the least gain where the interval lies above zero, the gain exceeds T and reaches the least
  gain; real and small where only the last fails; harm where the interval lies below zero and
  the loss exceeds T; noise otherwise.
- *Contrasts.* Fine-tuning under one set of pretrainings less under another reads against
  T = 2·√(SE² + (s²/2)·(1/n_a + 1/n_b)). A contrast moves the recipe off its reference where it
  exceeds T at one of the deciding budgets and lies below -T at none.
- *Null pairs.* The network from nothing does not read the backbone, so its pairs between
  campaigns must lie within a bound; one outside stops the reading until it is explained.

    uv run scripts/pretraining_screen_report.py --comparisons DIR
        --gain LABEL CAMPAIGNS [--gain ...] --contrast LABEL CAMPAIGNS_A CAMPAIGNS_B [...]
        --spread CAMPAIGN CAMPAIGN [--spread ...] --null CAMPAIGN CAMPAIGN [--null ...]
        --scratch CANDIDATE --tuned CANDIDATE [--budgets 20 50 200] [--deciding 50 200]
"""

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.contrast_reading import ContrastReading
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.outcome_level import OutcomeLevel
from emblema.evaluation.domain.statistics.paired_reading import PairedReading
from emblema.evaluation.domain.statistics.pretraining_spread import PretrainingSpread
from scripts.campaign_pairs_report import COMPARISONS, GROUPED_BY
from scripts.reporting import table

FLOOR = 0.014
LEAST_GAIN = 0.02
NULL_BOUND = 0.02
BUDGETS = ("20", "50", "200")
DECIDING = ("50", "200")
# The pairs report states its intervals at this level unless told otherwise.
LEVEL = 0.95


@dataclass(frozen=True, kw_only=True)
class Pair:
    """One row of ``comparisons.csv``: the candidate's gain over the control at one budget."""

    control_campaign: str
    control: str
    candidate_campaign: str
    candidate: str
    budget: str
    mean: float
    se: float
    low: float
    high: float

    @property
    def reading(self) -> PairedReading:
        return PairedReading(
            mean=self.mean,
            standard_error=self.se,
            interval=BootstrapInterval(low=self.low, high=self.high, level=LEVEL),
        )


def read_pairs(path: Path) -> list[Pair]:
    with path.open(newline="") as handle:
        return [
            Pair(
                control_campaign=row["control_campaign"],
                control=row["control"],
                candidate_campaign=row["candidate_campaign"],
                candidate=row["candidate"],
                budget=row["budget"],
                mean=float(row["seed_reduction_mean"]),
                se=float(row["seed_reduction_se"]),
                low=float(row["low"]),
                high=float(row["high"]),
            )
            for row in csv.DictReader(handle)
        ]


class Pairs:
    """The compared pairs, looked up by their sides."""

    def __init__(self, pairs: Iterable[Pair]) -> None:
        self._by_sides = {
            (p.control_campaign, p.control, p.candidate_campaign, p.candidate, p.budget): p
            for p in pairs
        }

    def find(
        self,
        control_campaign: str,
        control: str,
        candidate_campaign: str,
        candidate: str,
        budget: str,
    ) -> Pair:
        """The pair of these sides at this budget.

        Raises:
            ValueError: If it was not compared.
        """
        key = (control_campaign, control, candidate_campaign, candidate, budget)
        found = self._by_sides.get(key)
        if found is None:
            raise ValueError(
                f"no pair of {control} under {control_campaign} and {candidate} under "
                f"{candidate_campaign} at {budget} among the comparisons"
            )
        return found


def count(campaigns: str) -> int:
    return len(campaigns.split(GROUPED_BY))


@dataclass(frozen=True, kw_only=True)
class Gain:
    """Fine-tuning less the network from nothing under n pretrainings at one budget."""

    label: str
    pair: Pair
    n: int
    spread: PretrainingSpread

    @property
    def threshold(self) -> DetectionThreshold:
        return DetectionThreshold.of_gain(
            standard_error=self.pair.se, spread=self.spread.value, pretrainings=self.n
        )

    @property
    def level(self) -> OutcomeLevel:
        return self.pair.reading.level(self.threshold, least_gain=LEAST_GAIN)


@dataclass(frozen=True, kw_only=True)
class Contrast:
    """Fine-tuning under the pretrainings of B less under those of A at one budget."""

    label: str
    pair: Pair
    n_a: int
    n_b: int
    spread: PretrainingSpread
    decides: bool

    @property
    def reading(self) -> ContrastReading:
        return ContrastReading(
            reading=self.pair.reading,
            threshold=DetectionThreshold.of_contrast(
                standard_error=self.pair.se, spread=self.spread.value, sides=(self.n_a, self.n_b)
            ),
            decides=self.decides,
        )


@dataclass(frozen=True, kw_only=True)
class Screen:
    """Everything the declared rule reads: one spread, and each pair at each budget."""

    spread: PretrainingSpread
    nulls: tuple[Pair, ...]
    gains: tuple[Gain, ...]
    contrasts: tuple[Contrast, ...]
    deciding: tuple[str, ...]

    @property
    def nulls_hold(self) -> bool:
        return all(abs(p.mean) <= NULL_BOUND for p in self.nulls)

    def moved(self, label: str) -> bool:
        return ContrastReading.moves([c.reading for c in self.contrasts if c.label == label])


def screen(
    pairs: Pairs,
    *,
    scratch: str,
    tuned: str,
    gains: Sequence[tuple[str, str]],
    contrasts: Sequence[tuple[str, str, str]],
    spreads: Sequence[tuple[str, str]],
    nulls: Sequence[tuple[str, str]],
    budgets: Sequence[str],
    deciding: Sequence[str],
) -> Screen:
    """Every declared pair read by the rule.

    Raises:
        ValueError: If a declared pair was not compared, or no spread pair is given.
    """
    spread = PretrainingSpread.of(
        [pairs.find(a, tuned, b, tuned, budget).reading for budget in budgets for a, b in spreads],
        floor=FLOOR,
    )
    return Screen(
        spread=spread,
        nulls=tuple(
            pairs.find(a, scratch, b, scratch, budget) for budget in budgets for a, b in nulls
        ),
        gains=tuple(
            Gain(
                label=label,
                pair=pairs.find(campaigns, scratch, campaigns, tuned, budget),
                n=count(campaigns),
                spread=spread,
            )
            for budget in budgets
            for label, campaigns in gains
        ),
        contrasts=tuple(
            Contrast(
                label=label,
                pair=pairs.find(a, tuned, b, tuned, budget),
                n_a=count(a),
                n_b=count(b),
                spread=spread,
                decides=budget in deciding,
            )
            for budget in budgets
            for label, a, b in contrasts
        ),
        deciding=tuple(deciding),
    )


def _interval(pair: Pair) -> str:
    return f"[{pair.low:+.4f}; {pair.high:+.4f}]"


def render(result: Screen) -> str:
    nulls = [
        [
            p.budget,
            f"{p.control_campaign[:8]} → {p.candidate_campaign[:8]}",
            f"{p.mean:+.4f}",
            "within" if abs(p.mean) <= NULL_BOUND else "OUTSIDE",
        ]
        for p in result.nulls
    ]
    gains = [
        [
            g.pair.budget,
            g.label,
            str(g.n),
            f"{g.pair.mean:+.4f} ± {g.pair.se:.4f}",
            _interval(g.pair),
            f"{g.threshold.value:.4f}",
            g.level.value,
        ]
        for g in result.gains
    ]
    contrasts = [
        [
            c.pair.budget,
            c.label,
            f"{c.n_a} / {c.n_b}",
            f"{c.pair.mean:+.4f} ± {c.pair.se:.4f}",
            _interval(c.pair),
            f"{c.reading.threshold.value:.4f}",
            "above T" if c.reading.above else "below -T" if c.reading.below else "within ±T",
        ]
        for c in result.contrasts
    ]
    labels = list(dict.fromkeys(c.label for c in result.contrasts))
    verdicts = [
        f"{label}: {'moves the recipe' if result.moved(label) else 'leaves the reference'}"
        for label in labels
    ]
    sections = [
        f"**Spread between pretrainings**, over pairs and budgets: own {result.spread.own:.4f}, "
        f"s {result.spread.value:.4f} (floor {FLOOR}).",
        f"**Null pairs** (bound ±{NULL_BOUND}).\n\n"
        + table(["budget", "campaigns", "difference", ""], nulls),
    ]
    if not result.nulls_hold:
        sections.append("A null pair lies outside its bound: the reading stops here.")
        return "\n\n".join(sections)
    sections += [
        f"**Gains of fine-tuning over the network from nothing** (least gain {LEAST_GAIN}).\n\n"
        + table(["budget", "recipe", "n", "gain ± SE", "interval", "T", "level"], gains),
    ]
    if contrasts:
        sections += [
            "**Contrasts of fine-tuning between recipes.**\n\n"
            + table(["budget", "contrast", "n", "difference ± SE", "interval", "T", ""], contrasts),
            f"Deciding budgets {', '.join(result.deciding)}: " + "; ".join(verdicts) + ".",
        ]
    return "\n\n".join(sections)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--comparisons",
        type=Path,
        required=True,
        help=f"directory holding the {COMPARISONS} the pairs report wrote",
    )
    parser.add_argument("--scratch", required=True, help="the network from nothing")
    parser.add_argument("--tuned", required=True, help="fine-tuning of the backbone")
    parser.add_argument(
        "--gain", nargs=2, action="append", default=[], metavar=("LABEL", "CAMPAIGNS")
    )
    parser.add_argument(
        "--contrast",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABEL", "CAMPAIGNS_A", "CAMPAIGNS_B"),
    )
    parser.add_argument(
        "--spread", nargs=2, action="append", required=True, metavar=("CAMPAIGN", "CAMPAIGN")
    )
    parser.add_argument(
        "--null", nargs=2, action="append", default=[], metavar=("CAMPAIGN", "CAMPAIGN")
    )
    parser.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    parser.add_argument("--deciding", nargs="+", default=list(DECIDING))
    arguments = parser.parse_args(argv)
    if not arguments.gain and not arguments.contrast:
        parser.error("give a gain or a contrast to read")
    if not set(arguments.deciding) <= set(arguments.budgets):
        parser.error("the deciding budgets must be among the budgets read")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    result = screen(
        Pairs(read_pairs(arguments.comparisons / COMPARISONS)),
        scratch=arguments.scratch,
        tuned=arguments.tuned,
        gains=[(label, campaigns) for label, campaigns in arguments.gain],
        contrasts=[(label, a, b) for label, a, b in arguments.contrast],
        spreads=[(a, b) for a, b in arguments.spread],
        nulls=[(a, b) for a, b in arguments.null],
        budgets=arguments.budgets,
        deciding=arguments.deciding,
    )
    print(render(result))


if __name__ == "__main__":
    main()
