"""Which rate an arm trains at: the rate in force, unless another clearly does better.

An arm — fine-tuning a pretrained backbone, or the network trained from nothing — runs at several
rates on a fixed half of a task, at several budgets and seeds, and under several backbones where
it fine-tunes one. Its candidates are read by area under the ROC curve from every window's
answer, as exported by ``campaign_pairs_report.py`` into ``predictions.csv``.

The rule is the Evaluation context's (``RateChoice``); what is here is the reading it is fed.
The rate in force stays unless another rate's mean area over every repeat — each (campaign,
budget, seed) of the arm — exceeds it by more than one standard error of that difference. The
standard error comes from a paired bootstrap that draws the stays once per resample and scores
every repeat of both rates on that one draw, in the two strata the harness resamples by, so the
budgets, seeds and backbones that share stays share their noise rather than counting as
independent. Where several rates clear the bar, the one with the largest gain is chosen. A
choice at either end of the grid is named as such: the best rate may lie beyond it.

The rule replaces the rule of one standard error over repeats, which needs repeats that score
different units; on a fixed half every repeat scores the same stays.

    uv run scripts/rate_choice_report.py --predictions DIR --rate-in-force 0.001
        --arm LABEL CAMPAIGN[+CAMPAIGN...] IN_FORCE RATE_CANDIDATE [RATE_CANDIDATE ...]
        [--arm ...] [--budgets 20 50 200] [--out DIR]
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from emblema.evaluation.domain.statistics.rate_choice import RateChoice, RateOption
from scripts.campaign_pairs_report import GROUPED_BY, PREDICTIONS, Answers, Side, read_predictions
from scripts.reporting import table

CHOICES = "choices.csv"
CHOICE_COLUMNS = (
    "arm",
    "candidate",
    "rate",
    "repeats",
    "area",
    "gain",
    "standard_error",
    "clears",
    "chosen",
)
RATE_KEY = "learning_rate"
BUDGETS = ("20", "50", "200")


@dataclass(frozen=True, kw_only=True)
class Arm:
    """One arm's candidates, the rate in force first, read under the named campaigns."""

    label: str
    campaigns: tuple[str, ...]
    in_force: str
    others: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class RateReading:
    """One rate of an arm beside the rate in force, over every repeat of the arm."""

    arm: str
    candidate: str
    rate: float
    repeats: int
    area: float
    gain: float
    standard_error: float

    @property
    def option(self) -> RateOption:
        return RateOption(rate=self.rate, gain=self.gain, standard_error=self.standard_error)


@dataclass(frozen=True, kw_only=True)
class ArmChoice:
    """What the rule chose for an arm, from the readings of every rate on its grid."""

    arm: str
    in_force: RateReading
    others: tuple[RateReading, ...]

    @property
    def rule(self) -> RateChoice:
        return RateChoice(
            in_force=self.in_force.rate, others=tuple(each.option for each in self.others)
        )

    @property
    def chosen(self) -> RateReading:
        rate = self.rule.chosen
        return next(each for each in (self.in_force, *self.others) if each.rate == rate)

    @property
    def at_edge(self) -> bool:
        return self.rule.at_edge


def rate_of(candidate: str, in_force: float) -> float:
    """The rate a candidate states in its parameters, or the rate in force where it states none.

    Raises:
        ValueError: If the candidate states its rate other than as a number.
    """
    _, _, parameters = candidate.partition("@")
    for parameter in parameters.split(",") if parameters else ():
        key, _, value = parameter.partition("=")
        if key == RATE_KEY:
            try:
                return float(value)
            except ValueError as error:
                raise ValueError(f"{candidate} states its rate as {value!r}") from error
    return in_force


def repeats_of(arm: Arm, candidate: str, budgets: Sequence[str]) -> list[Side]:
    """Every (campaign, budget) side of a candidate of the arm, in one order for every rate."""
    return [Side(campaign, candidate, budget) for campaign in arm.campaigns for budget in budgets]


def rankings_of(
    answers: Answers, arm: Arm, candidate: str, budgets: Sequence[str], seeds: Sequence[int]
) -> tuple[WindowRanking, ...]:
    return tuple(
        answers.ranking(side, seed)
        for side in repeats_of(arm, candidate, budgets)
        for seed in seeds
    )


def seeds_of(answers: Answers, arm: Arm, budgets: Sequence[str]) -> list[int]:
    """The seeds every side of every rate of the arm ran under.

    Raises:
        ValueError: If some side was not run, or ran under other seeds than the rest.
    """
    found = answers.sides()
    seeds: list[int] | None = None
    for candidate in (arm.in_force, *arm.others):
        for side in repeats_of(arm, candidate, budgets):
            if side not in found:
                raise ValueError(f"no answer of {side} among the exported predictions")
            if seeds is None:
                seeds = found[side]
            elif found[side] != seeds:
                raise ValueError(f"{side} ran under seeds {found[side]}, the arm under {seeds}")
    if seeds is None:
        raise ValueError(f"arm {arm.label} names no candidate")
    return seeds


def choose(
    answers: Answers,
    arm: Arm,
    *,
    rate_in_force: float,
    budgets: Sequence[str],
    bootstrap: PairedUnitBootstrap,
) -> ArmChoice:
    """Each rate of the arm read against the rate in force, and the rate the rule keeps.

    Raises:
        ValueError: If a side of the arm is missing or ran under other seeds, or two repeats
            scored different stays, which a fixed half never does.
    """
    seeds = seeds_of(answers, arm, budgets)
    control = rankings_of(answers, arm, arm.in_force, budgets, seeds)

    def reading(candidate: str, gain: float, error: float) -> RateReading:
        rankings = rankings_of(answers, arm, candidate, budgets, seeds)
        return RateReading(
            arm=arm.label,
            candidate=candidate,
            rate=rate_of(candidate, rate_in_force),
            repeats=len(rankings),
            area=fmean(ranking.auroc for ranking in rankings),
            gain=gain,
            standard_error=error,
        )

    others = []
    for candidate in arm.others:
        paired = PairedUnitRankings(
            control=control, candidate=rankings_of(answers, arm, candidate, budgets, seeds)
        )
        others.append(reading(candidate, paired.reduction, stdev(bootstrap.reductions(paired))))
    return ArmChoice(arm=arm.label, in_force=reading(arm.in_force, 0.0, 0.0), others=tuple(others))


def write_choices(path: Path, choices: Sequence[ArmChoice]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHOICE_COLUMNS)
        for choice in choices:
            for each in (choice.in_force, *choice.others):
                writer.writerow(
                    (
                        each.arm,
                        each.candidate,
                        repr(each.rate),
                        each.repeats,
                        repr(each.area),
                        repr(each.gain),
                        repr(each.standard_error),
                        each is not choice.in_force and each.option.clears,
                        each is choice.chosen,
                    )
                )


def render(choices: Sequence[ArmChoice]) -> str:
    sections = []
    for choice in choices:
        rows = [
            [
                f"{each.rate:g}" + (" (in force)" if each is choice.in_force else ""),
                str(each.repeats),
                f"{each.area:.4f}",
                "—" if each is choice.in_force else f"{each.gain:+.4f}",
                "—" if each is choice.in_force else f"{each.standard_error:.4f}",
                "—" if each is choice.in_force else ("yes" if each.option.clears else "no"),
            ]
            for each in sorted((choice.in_force, *choice.others), key=lambda r: r.rate)
        ]
        chosen = choice.chosen
        verdict = f"chosen rate: {chosen.rate:g}" + (
            " (the rate in force)" if chosen is choice.in_force else f" ({chosen.candidate})"
        )
        if choice.at_edge:
            verdict += " — at an edge of the grid"
        sections.append(
            f"**{choice.arm}**\n\n"
            + table(["rate", "repeats", "mean area", "gain", "standard error", "clears"], rows)
            + f"\n\n{verdict}."
        )
    return "\n\n".join(sections)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=f"directory holding the {PREDICTIONS} the pairs report exported",
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        action="append",
        required=True,
        metavar="LABEL CAMPAIGNS IN_FORCE CANDIDATE",
        help=f"an arm: its label, its campaigns joined by {GROUPED_BY!r}, the candidate at the "
        "rate in force, then the candidates at the other rates",
    )
    parser.add_argument(
        "--rate-in-force",
        type=float,
        required=True,
        help="the rate a candidate stating none runs at",
    )
    parser.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1, help="seed of the resampling")
    parser.add_argument("--out", type=Path, help=f"directory {CHOICES} is written to")
    arguments = parser.parse_args(argv)
    for arm in arguments.arm:
        if len(arm) < 4:
            parser.error(
                "--arm needs a label, its campaigns, the candidate in force and one other rate"
            )
    return arguments


def arms_of(arguments: argparse.Namespace) -> list[Arm]:
    return [
        Arm(
            label=label,
            campaigns=tuple(campaigns.split(GROUPED_BY)),
            in_force=in_force,
            others=tuple(others),
        )
        for label, campaigns, in_force, *others in arguments.arm
    ]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    answers = Answers(read_predictions(arguments.predictions / PREDICTIONS))
    bootstrap = PairedUnitBootstrap(resamples=arguments.resamples, seed=arguments.seed)
    choices = [
        choose(
            answers,
            arm,
            rate_in_force=arguments.rate_in_force,
            budgets=arguments.budgets,
            bootstrap=bootstrap,
        )
        for arm in arms_of(arguments)
    ]
    if arguments.out is not None:
        arguments.out.mkdir(parents=True, exist_ok=True)
        write_choices(arguments.out / CHOICES, choices)
    print(render(choices))


if __name__ == "__main__":
    main()
