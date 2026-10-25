"""Whether fine-tuning gains more on the stays its backbone read in pretraining than on the rest.

A backbone pretrained on a share of a task's tuning side has read, without labels, some of the
stays a campaign later scores on its held-out half. A gain from fine-tuning that comes from
having seen those stays would be larger on them than on the stays the backbone never read. This
script splits each scored half into the two and reads fine-tuning's gain over the network from
nothing on each, from every window's answer as ``campaign_pairs_report.py`` exported it into
``predictions.csv``.

Each backbone is read beside the control of its own campaign, every seed a repeat pooled by the
mean of the areas, as the harness pools them. A backbone's gain on the stays it read less its
gain on the stays it left is the transduction; the mean over backbones is read per budget. The
interval comes from one bootstrap that draws the scored stays once per resample, in the two
strata the harness resamples by, and splits the draw into read and unread for every backbone, so
backbones that read the same stays share their noise. The interval holds the seeds as they ran,
so it cannot see how far other draws of labels would move the gap. Descriptive: no rule is judged
by it.

    uv run --env-file .env.r2 scripts/transduction_report.py --predictions DIR --corpus CORPUS
        --control CANDIDATE --candidate CANDIDATE
        --pair LABEL CAMPAIGN BACKBONE [--pair ...] [--budgets 20 50 200] [--out DIR]
"""

import argparse
import csv
import random
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from uuid import UUID

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from scripts.campaign_pairs_report import PREDICTIONS, Answers, Side, read_predictions
from scripts.reporting import table

TRANSDUCTION = "transduction.csv"
TRANSDUCTION_COLUMNS = ("label", "budget", "read", "unread", "gain_read", "gain_unread", "gap")
MEAN = "mean"
BUDGETS = ("20", "50", "200")


@dataclass(frozen=True, kw_only=True)
class Pair:
    """One backbone's fine-tuning beside its campaign's control, and the stays it read."""

    label: str
    campaign: str
    seen: frozenset[str]


@dataclass(frozen=True, kw_only=True)
class Split:
    """One backbone's gain at one budget, on the scored stays it read and on those it left."""

    label: str
    budget: str
    read: int
    unread: int
    gain_read: float
    gain_unread: float

    @property
    def gap(self) -> float:
        """How much more fine-tuning gains on the stays the backbone read."""
        return self.gain_read - self.gain_unread


@dataclass(frozen=True, kw_only=True)
class BudgetReading:
    """Every backbone's split at one budget, and their mean gap with its interval."""

    budget: str
    splits: tuple[Split, ...]
    low: float
    high: float
    standard_error: float

    @property
    def gap(self) -> float:
        return fmean(split.gap for split in self.splits)

    @property
    def gain_read(self) -> float:
        return fmean(split.gain_read for split in self.splits)

    @property
    def gain_unread(self) -> float:
        return fmean(split.gain_unread for split in self.splits)


def paired_of(
    answers: Answers, pair: Pair, control: str, candidate: str, budget: str
) -> PairedUnitRankings:
    """Fine-tuning against the control of the pair's campaign at one budget, over every seed.

    Raises:
        ValueError: If either side was not exported, or the two ran under other seeds.
    """
    sides = answers.sides()
    control_side = Side(pair.campaign, control, budget)
    candidate_side = Side(pair.campaign, candidate, budget)
    for side in (control_side, candidate_side):
        if side not in sides:
            raise ValueError(f"no answer of {side} among the exported predictions")
    seeds = sides[control_side]
    if sides[candidate_side] != seeds:
        raise ValueError(f"{candidate_side} ran under seeds {sides[candidate_side]}, not {seeds}")
    return PairedUnitRankings(
        control=tuple(answers.ranking(control_side, seed) for seed in seeds),
        candidate=tuple(answers.ranking(candidate_side, seed) for seed in seeds),
    )


def read_and_unread(
    paired: PairedUnitRankings, seen: Collection[str]
) -> tuple[list[int], list[int]]:
    """Indices of the scored stays the backbone read, then of those it left.

    Raises:
        ValueError: If either part is empty, so there is no gain to set beside the other.
    """
    read = [index for index, unit in enumerate(paired.units) if unit.value in seen]
    unread = [index for index, unit in enumerate(paired.units) if unit.value not in seen]
    if not read or not unread:
        raise ValueError(
            f"the backbone read {len(read)} of the {len(paired.units)} scored stays: "
            "a gap needs stays on both sides"
        )
    return read, unread


def transduction(
    answers: Answers,
    pairs: Sequence[Pair],
    *,
    control: str,
    candidate: str,
    budgets: Sequence[str],
    bootstrap: PairedUnitBootstrap,
) -> list[BudgetReading]:
    """Every backbone's gap at every budget, and the mean gap with its interval per budget.

    Raises:
        ValueError: If a side is missing, the pairs scored different stays at a budget, or a
            backbone read none or all of the scored stays.
    """
    readings = []
    for budget in budgets:
        pairings = [paired_of(answers, pair, control, candidate, budget) for pair in pairs]
        units = pairings[0].units
        if any(paired.units != units for paired in pairings):
            raise ValueError(f"the pairs scored different stays at {budget}")
        parts = [
            read_and_unread(paired, pair.seen) for paired, pair in zip(pairings, pairs, strict=True)
        ]
        splits = tuple(
            Split(
                label=pair.label,
                budget=budget,
                read=len(read),
                unread=len(unread),
                gain_read=paired.reduction_over(read),
                gain_unread=paired.reduction_over(unread),
            )
            for pair, paired, (read, unread) in zip(pairs, pairings, parts, strict=True)
        )
        gaps = sorted(_resampled_gaps(pairings, pairs, bootstrap))
        interval = bootstrap.interval_of(gaps)
        readings.append(
            BudgetReading(
                budget=budget,
                splits=splits,
                low=interval.low,
                high=interval.high,
                standard_error=stdev(gaps),
            )
        )
    return readings


def _resampled_gaps(
    pairings: Sequence[PairedUnitRankings], pairs: Sequence[Pair], bootstrap: PairedUnitBootstrap
) -> list[float]:
    """The mean gap over backbones in every resample, the stays drawn once for all of them.

    The draw is the one ``PairedUnitBootstrap.reductions`` makes — each stratum with replacement,
    at its own size, off one seeded generator — repeated here because every draw is split twice.
    """
    strata = pairings[0].strata
    draws = random.Random(bootstrap.seed)
    seen = [
        {index for index, unit in enumerate(paired.units) if unit.value in pair.seen}
        for paired, pair in zip(pairings, pairs, strict=True)
    ]
    gaps = []
    for _ in range(bootstrap.resamples):
        picks = [pick for stratum in strata for pick in draws.choices(stratum, k=len(stratum))]
        gaps.append(
            fmean(
                paired.reduction_over([pick for pick in picks if pick in read])
                - paired.reduction_over([pick for pick in picks if pick not in read])
                for paired, read in zip(pairings, seen, strict=True)
            )
        )
    return gaps


def write_transduction(path: Path, readings: Sequence[BudgetReading]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRANSDUCTION_COLUMNS)
        for reading in readings:
            for split in reading.splits:
                writer.writerow(
                    (
                        split.label,
                        split.budget,
                        split.read,
                        split.unread,
                        repr(split.gain_read),
                        repr(split.gain_unread),
                        repr(split.gap),
                    )
                )
            writer.writerow(
                (
                    MEAN,
                    reading.budget,
                    "",
                    "",
                    repr(reading.gain_read),
                    repr(reading.gain_unread),
                    repr(reading.gap),
                )
            )


def render(readings: Sequence[BudgetReading]) -> str:
    rows = [
        [
            reading.budget,
            f"{reading.gain_read:+.4f}",
            f"{reading.gain_unread:+.4f}",
            f"{reading.gap:+.4f}",
            f"[{reading.low:+.4f}; {reading.high:+.4f}]",
            f"{reading.standard_error:.4f}",
        ]
        for reading in readings
    ]
    detail = [
        [
            split.budget,
            split.label,
            f"{split.read} / {split.unread}",
            f"{split.gain_read:+.4f}",
            f"{split.gain_unread:+.4f}",
            f"{split.gap:+.4f}",
        ]
        for reading in readings
        for split in reading.splits
    ]
    return "\n\n".join(
        [
            "**Fine-tuning's gain in area over the network from nothing, on the scored stays "
            "the backbone read and on those it left, the mean over backbones.**",
            table(
                ["budget", "gain read", "gain unread", "gap", "interval", "standard error"], rows
            ),
            table(
                ["budget", "backbone", "read / unread", "gain read", "gain unread", "gap"], detail
            ),
        ]
    )


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
    parser.add_argument("--corpus", required=True, help="the corpus the task's stays belong to")
    parser.add_argument("--control", required=True, help="the network from nothing")
    parser.add_argument("--candidate", required=True, help="fine-tuning of the backbone")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CAMPAIGN", "BACKBONE"),
        help="a backbone's campaign and the backbone it fine-tuned",
    )
    parser.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1, help="seed of the resampling")
    parser.add_argument("--out", type=Path, help=f"directory {TRANSDUCTION} is written to")
    arguments = parser.parse_args(argv)
    for _, _, backbone in arguments.pair:
        try:
            UUID(backbone)
        except ValueError:
            parser.error(f"--pair: {backbone!r} is not a backbone's identifier")
    return arguments


def seen_by(backbone: str, corpus: str) -> frozenset[str]:  # pragma: no cover - the registry
    """The training units of ``corpus`` the backbone's run read, as its share chose them.

    Raises:
        SystemExit: If the backbone did not read the corpus.
    """
    from emblema.catalog.contracts.published_corpus_manifest_json import (
        PublishedCorpusManifestJson,
    )
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_engine, configured_store
    from emblema.pretraining.adapters.persistence.backbone_repository import (
        SqlAlchemyBackboneRepository,
    )
    from emblema.pretraining.domain.identifiers import BackboneId

    settings = Settings()
    found = SqlAlchemyBackboneRepository(configured_engine(settings)).get(
        BackboneId(UUID(backbone))
    )
    stated = next((given for given in found.inputs if given.corpus == corpus), None)
    if stated is None:
        raise SystemExit(f"backbone {backbone} did not read {corpus}")
    manifest = PublishedCorpusManifestJson().decode(configured_store(settings).get(stated.manifest))
    return frozenset(found.configuration.corpus_share_of(corpus).select(manifest.training_units))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    pairs = [
        Pair(label=label, campaign=campaign, seen=seen_by(backbone, arguments.corpus))
        for label, campaign, backbone in arguments.pair
    ]
    readings = transduction(
        Answers(read_predictions(arguments.predictions / PREDICTIONS)),
        pairs,
        control=arguments.control,
        candidate=arguments.candidate,
        budgets=arguments.budgets,
        bootstrap=PairedUnitBootstrap(resamples=arguments.resamples, seed=arguments.seed),
    )
    if arguments.out is not None:
        arguments.out.mkdir(parents=True, exist_ok=True)
        write_transduction(arguments.out / TRANSDUCTION, readings)
    print(render(readings))


if __name__ == "__main__":
    main()
