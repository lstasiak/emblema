"""How often the replacement rule replaces a configuration whose true gain is known.

A comparison on the intensive-care task is read by two conditions: the paired interval of the
difference in area, pooled over seeds, lies above zero, and the mean over seeds exceeds twice its
standard error — which, between backbones, is widened by the spread s that pretraining alone
gives a difference, to twice the root of SE² + s². How often that rule fires at a true gain
depends on how many seeds ran and how many stays were scored. This script measures the noise
each of those carries from answers already exported by ``campaign_pairs_report.py`` and
simulates the rule over designs of seeds and stays, for the pairs it is given, at every budget
both sides of a pair hold.

The stays a design scores are drawn from the exported ones, with replacement, in the two strata of
the outcome. Every draw gives each seed's difference in area and the standard error of the
difference pooled over seeds from placement values (DeLong), the normal approximation of the
stratified bootstrap the verdict resamples by. From the draws three parts of the variance are
read — a seed's own effect, the shared effect of which stays were scored, and their interaction —
and a normal model of them answers for any number of seeds. At the seeds that ran, the model is
checked against applying the rule to the draws themselves.

Where every seed of both sides scored the same labels, as at the budget of every labelled unit,
the seeds of a closed-form reading do not differ at all. The seed condition then has nothing to
read and is left out: the rule is the interval and the threshold, and the components say so with
a seed effect of zero.

    uv run scripts/comparison_power_report.py --predictions CSV [--predictions CSV] --out DIR
        --pair KIND LABEL CONTROL_CAMPAIGN CONTROL CANDIDATE_CAMPAIGN CANDIDATE [--pair ...]
        [--budgets B ...] [--stays N ...] [--gains G ...] [--seeds K ...]
        [--seed-gap KIND VALUE ...] [--check-predictions CSV --check-comparisons CSV]

A pair is read at every budget both of its sides were exported at, or at ``--budgets`` alone.
``--stays`` defaults to every stay the budget scored; ``--seed-gap`` names, per kind of pair, the
spread s between backbones the seed condition is widened by for that kind, at every budget as
``VALUE`` or at one as ``BUDGET=VALUE``. Writes
``components.csv``, ``power.csv`` and, with the check, ``check.csv``, and prints the tables a
note pastes.
"""

import argparse
import csv
import statistics
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import inf, sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting import table

COMPONENTS, POWER, CHECK = "components.csv", "power.csv", "check.csv"

SEEDS = (5, 10, 20, 30)
GAINS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)
Z_95 = 1.959963984540054

Floats = NDArray[np.float64]
Counts = NDArray[np.int64]


class SideKey(NamedTuple):
    """One candidate of one campaign at one budget."""

    campaign: str
    candidate: str
    budget: str


@dataclass(frozen=True)
class Answers:
    """Every seed's answer on every stay of one side, the stays in one order shared by all sides.

    Attributes:
        scores: ``[seeds, stays]``.
        outcome: ``[stays]``, true where the stay holds a death.
        units: The stays, in the order of both arrays.
    """

    scores: Floats
    outcome: NDArray[np.bool_]
    units: tuple[str, ...]


class PairSpec(NamedTuple):
    """A candidate against a control, as the command line names them, read at every budget."""

    kind: str
    name: str
    control_campaign: str
    control: str
    candidate_campaign: str
    candidate: str


class Pair(NamedTuple):
    """A candidate against a control, read on the same stays and seeds; positive favours it."""

    kind: str
    name: str
    budget: str
    control: SideKey
    candidate: SideKey


def read_answers(paths: Sequence[Path], campaigns: set[str]) -> dict[SideKey, Answers]:
    """The answers of the named campaigns, one window per stay, stays sorted by key.

    A campaign exported to more than one file is read from the first that holds it.

    Raises:
        ValueError: If a stay holds more than one window, a side's seeds scored other stays, or
            two sides disagree on a stay's outcome.
    """
    rows: dict[SideKey, dict[int, dict[str, float]]] = {}
    outcome: dict[str, bool] = {}
    for path in paths:
        _read_file(path, campaigns - {key.campaign for key in rows}, rows, outcome)
    found = {}
    for key, seeds in rows.items():
        units = sorted(next(iter(seeds.values())))
        if any(sorted(each) != units for each in seeds.values()):
            raise ValueError(f"the seeds of {key} scored different stays")
        found[key] = Answers(
            scores=np.array([[seeds[s][u] for u in units] for s in sorted(seeds)]),
            outcome=np.array([outcome[u] for u in units]),
            units=tuple(units),
        )
    return found


def _read_file(
    path: Path,
    campaigns: set[str],
    rows: dict[SideKey, dict[int, dict[str, float]]],
    outcome: dict[str, bool],
) -> None:
    with path.open(newline="") as stream:
        for line in csv.DictReader(stream):
            if line["campaign"] not in campaigns:
                continue
            key = SideKey(line["campaign"], line["candidate"], line["budget"])
            per_seed = rows.setdefault(key, {}).setdefault(int(line["seed"]), {})
            unit = line["unit"]
            if unit in per_seed:
                raise ValueError(f"{key} scores more than one window of {unit}")
            per_seed[unit] = float(line["predicted"])
            dies = float(line["target"]) == 1.0
            if outcome.setdefault(unit, dies) != dies:
                raise ValueError(f"the exported answers disagree on the outcome of {unit}")


def pairs_of(
    specs: Sequence[PairSpec], answers: Mapping[SideKey, Answers], budgets: Sequence[str] = ()
) -> list[Pair]:
    """Each named pair at every budget both of its sides hold, or at ``budgets`` alone.

    Raises:
        ValueError: If a named budget is one a side was not exported at, or a pair shares no
            budget at all.
    """
    found = []
    for spec in specs:
        held = {
            key.budget
            for key in answers
            if (key.campaign, key.candidate) == (spec.control_campaign, spec.control)
        } & {
            key.budget
            for key in answers
            if (key.campaign, key.candidate) == (spec.candidate_campaign, spec.candidate)
        }
        wanted = budgets or sorted(held, key=budget_order)
        for budget in wanted:
            if budget not in held:
                raise ValueError(f"{spec.name} was not exported at {budget} on both sides")
            found.append(
                Pair(
                    spec.kind,
                    spec.name,
                    budget,
                    SideKey(spec.control_campaign, spec.control, budget),
                    SideKey(spec.candidate_campaign, spec.candidate, budget),
                )
            )
        if not wanted:
            raise ValueError(f"{spec.name} shares no budget between its sides")
    return found


def budget_order(budget: str) -> float:
    """Budgets in the order of their size, the budget of every labelled unit last."""
    try:
        return float(budget)
    except ValueError:
        return inf


class StayDraws:
    """Draws of a design's stays from the exported ones, as a multiplicity per stay.

    Each draw takes the design's count from either stratum of the outcome, with replacement, at
    the exported stays' prevalence.
    """

    def __init__(self, outcome: NDArray[np.bool_], stays: int, draws: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.outcome = outcome
        positive = np.flatnonzero(outcome)
        negative = np.flatnonzero(~outcome)
        deaths = round(stays * len(positive) / len(outcome))
        self.counts: Counts = np.zeros((draws, len(outcome)), dtype=np.int64)
        for members, size in ((positive, deaths), (negative, stays - deaths)):
            picks = members[rng.integers(0, len(members), size=(draws, size))]
            flat = (picks + np.arange(draws)[:, None] * len(outcome)).ravel()
            self.counts += np.bincount(flat, minlength=draws * len(outcome)).reshape(
                draws, len(outcome)
            )


@dataclass(frozen=True)
class DrawnSide:
    """One side read on every draw.

    Attributes:
        areas: ``[draws, seeds]``, each seed's area on the drawn stays.
        placements: ``[draws, stays]``, each stay's placement averaged over the seeds: for a
            death, the share of drawn survivors it outranks; for a survivor, the share of drawn
            deaths that outrank it; ties count half.
    """

    areas: Floats
    placements: Floats


def drawn_side(answers: Answers, counts: Counts) -> DrawnSide:
    """Areas and placements of every seed of a side, on every draw at once."""
    outcome = answers.outcome
    deaths = counts[:, outcome].sum(axis=1)[:, None].astype(np.float64)
    survivors = counts[:, ~outcome].sum(axis=1)[:, None].astype(np.float64)
    areas = np.empty((counts.shape[0], answers.scores.shape[0]))
    placements = np.zeros(counts.shape, dtype=np.float64)
    for seed, scores in enumerate(answers.scores):
        order = np.argsort(scores, kind="stable")
        ranked = scores[order]
        starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
        group = np.cumsum(np.r_[True, ranked[1:] != ranked[:-1]]) - 1
        weights = counts[:, order].astype(np.float64)
        dies = outcome[order]
        tied_survivors = np.add.reduceat(weights * ~dies, starts, axis=1)
        tied_deaths = np.add.reduceat(weights * dies, starts, axis=1)
        below = np.cumsum(tied_survivors, axis=1) - tied_survivors
        above = deaths - np.cumsum(tied_deaths, axis=1)
        outranks = (below + 0.5 * tied_survivors)[:, group] / survivors
        outranked = (above + 0.5 * tied_deaths)[:, group] / deaths
        placement = np.where(dies, outranks, outranked)
        areas[:, seed] = (weights * dies * outranks).sum(axis=1) / deaths[:, 0]
        placements[:, order] += placement
    placements /= answers.scores.shape[0]
    return DrawnSide(areas=areas, placements=placements)


def pooled_error(
    control: DrawnSide, candidate: DrawnSide, counts: Counts, outcome: NDArray[np.bool_]
) -> Floats:
    """The standard error over stays of the difference in area pooled over seeds, per draw."""
    difference = candidate.placements - control.placements
    variance = np.zeros(counts.shape[0])
    for stratum in (outcome, ~outcome):
        weights = counts[:, stratum].astype(np.float64)
        values = difference[:, stratum]
        size = weights.sum(axis=1)
        centre = (weights * values).sum(axis=1) / size
        spread = (weights * (values - centre[:, None]) ** 2).sum(axis=1) / (size - 1)
        variance += spread / size
    return np.sqrt(variance)


@dataclass(frozen=True)
class Components:
    """The parts of a seed's difference in area at one number of stays scored.

    Attributes:
        seed: Standard deviation of a seed's own effect, the same on every set of stays.
        stays: Standard deviation of the effect of which stays were scored, shared by every seed.
        interaction: Standard deviation of what a seed and the stays scored make together.
        effects: The seeds' own effects as measured, about their mean.
        gap: The difference over every exported stay and seed.
        seed_error: The standard error of the mean over seeds, as a reading of the seeds that
            ran would state it: the seeds' spread within a draw over the root of their count,
            averaged over the draws.
    """

    seed: float
    stays: float
    interaction: float
    effects: Floats
    gap: float
    seed_error: float

    @classmethod
    def of(cls, differences: Floats, gap: float) -> "Components":
        """Read from ``[draws, seeds]`` differences of the same seeds on different stays."""
        seeds = differences.shape[1]
        per_seed = float(differences.var(axis=0, ddof=1).mean())
        of_mean = float(differences.mean(axis=1).var(ddof=1))
        interaction = max((per_seed - of_mean) * seeds / (seeds - 1), 0.0)
        effects = differences.mean(axis=0) - differences.mean()
        return cls(
            seed=sqrt(float(effects.var(ddof=1))),
            stays=sqrt(max(per_seed - interaction, 0.0)),
            interaction=sqrt(interaction),
            effects=effects,
            gap=gap,
            seed_error=float((differences.std(axis=1, ddof=1) / sqrt(seeds)).mean()),
        )

    def interval_error(self, seeds: int) -> float:
        return sqrt(self.stays**2 + self.interaction**2 / seeds)


def replaces(
    differences: Floats, interval_error: Floats | float, threshold: float | None
) -> NDArray[np.bool_]:
    """The rule, applied to ``[readings, seeds]`` differences with each reading's interval error.

    ``threshold`` is the spread s between backbones, which widens the seed condition to twice the
    root of SE² + s²; ``None`` for a comparison within one backbone. A reading whose seeds do not
    differ at all has no seed condition of its own to meet: its seeds scored the same labels and
    answered alike, so their spread says nothing of a draw of labels, and only s remains.
    """
    seeds = differences.shape[1]
    mean = differences.mean(axis=1)
    seed_error = differences.std(axis=1, ddof=1) / sqrt(seeds)
    spread = 0.0 if threshold is None else threshold
    widened = np.sqrt(seed_error**2 + spread**2)
    fires: NDArray[np.bool_] = (mean - Z_95 * np.asarray(interval_error) > 0) & (
        (widened == 0.0) | (mean > 2 * widened)
    )
    return fires


def modelled(
    components: Components,
    seeds: int,
    gain: float,
    threshold: float | None,
    readings: int,
    rng: np.random.Generator,
    *,
    measured_seeds: bool = False,
) -> float:
    """The share of normal readings in which the rule replaces.

    With ``measured_seeds`` the seeds' own effects are the ones measured rather than drawn, as
    the draws of stays hold them.
    """
    if measured_seeds:
        effects = np.broadcast_to(components.effects, (readings, len(components.effects)))
    else:
        effects = rng.normal(0.0, components.seed, size=(readings, seeds))
    width = effects.shape[1]
    differences = (
        gain
        + effects
        + rng.normal(0.0, components.stays, size=(readings, 1))
        + rng.normal(0.0, components.interaction, size=(readings, width))
    )
    return float(replaces(differences, components.interval_error(width), threshold).mean())


def direct(
    differences: Floats, errors: Floats, gap: float, gain: float, threshold: float | None
) -> float:
    """The share of draws in which the rule replaces, the differences shifted to a true ``gain``."""
    return float(replaces(differences - gap + gain, errors, threshold).mean())


class PowerRow(NamedTuple):
    kind: str
    pair: str
    budget: str
    stays: int
    seeds: int
    gain: float
    threshold: float | None
    source: str
    power: float


class ComponentRow(NamedTuple):
    kind: str
    pair: str
    budget: str
    stays: int
    gap: float
    seed: float
    stays_part: float
    interaction: float
    mean_interval_error: float
    seed_error: float


@dataclass(frozen=True)
class Design:
    """What the simulation runs over: stays scored, seeds modelled, gains and thresholds.

    Attributes:
        stays: Counts of stays a design scores; empty for every stay the budget scored.
        seeds: Counts of seeds the normal model answers for.
        gains: True gains the rule is read at; zero is the rate of false replacement.
        seed_gaps: Per kind of pair, the spreads s between backbones its seed condition is
            widened by, each at one budget or, named by ``None``, at every budget; a kind not
            named compares within one backbone.
    """

    stays: tuple[int, ...] = ()
    seeds: tuple[int, ...] = SEEDS
    gains: tuple[float, ...] = GAINS
    seed_gaps: Mapping[str, tuple[tuple[str | None, float], ...]] = field(default_factory=dict)

    def thresholds_of(self, kind: str, budget: str) -> tuple[float | None, ...]:
        found = tuple(
            value for at, value in self.seed_gaps.get(kind, ()) if at is None or at == budget
        )
        return found or (None,)

    def stays_of(self, scored: int) -> tuple[int, ...]:
        return self.stays or (scored,)


DEFAULT_DESIGN = Design()


def simulate(
    answers: Mapping[SideKey, Answers],
    pairs: Sequence[Pair],
    *,
    draws: int,
    readings: int,
    seed: int,
    design: Design = DEFAULT_DESIGN,
) -> tuple[list[ComponentRow], list[PowerRow]]:
    """Components and power for every pair at every number of stays and seeds.

    Raises:
        ValueError: If two sides of a budget scored different stays.
    """
    components: list[ComponentRow] = []
    power: list[PowerRow] = []
    rng = np.random.default_rng(seed)
    budgets = sorted({p.budget for p in pairs}, key=budget_order)
    for index, budget in enumerate(budgets):
        at_budget = [p for p in pairs if p.budget == budget]
        first = answers[at_budget[0].control]
        for side in {side for p in at_budget for side in (p.control, p.candidate)}:
            if answers[side].units != first.units:
                raise ValueError(f"{side} scored other stays than {at_budget[0].control}")
        outcome = first.outcome
        for stays in design.stays_of(len(outcome)):
            drawn = StayDraws(outcome, stays, draws, seed=seed + stays + 7919 * index)
            read = _read_pairs(answers, at_budget, drawn.counts, outcome)
            for p in at_budget:
                differences, errors = read[p]
                gap = population_gap(answers[p.control], answers[p.candidate])
                parts = Components.of(differences, gap)
                components.append(
                    ComponentRow(
                        p.kind,
                        p.name,
                        budget,
                        stays,
                        gap,
                        parts.seed,
                        parts.stays,
                        parts.interaction,
                        float(errors.mean()),
                        parts.seed_error,
                    )
                )
                for threshold in design.thresholds_of(p.kind, budget):
                    for gain in design.gains:
                        row = (p.kind, p.name, budget, stays)
                        power.append(
                            PowerRow(
                                *row,
                                differences.shape[1],
                                gain,
                                threshold,
                                "direct",
                                direct(differences, errors, gap, gain, threshold),
                            )
                        )
                        power.append(
                            PowerRow(
                                *row,
                                differences.shape[1],
                                gain,
                                threshold,
                                "measured seeds",
                                modelled(
                                    parts, 0, gain, threshold, readings, rng, measured_seeds=True
                                ),
                            )
                        )
                        for count in design.seeds:
                            power.append(
                                PowerRow(
                                    *row,
                                    count,
                                    gain,
                                    threshold,
                                    "model",
                                    modelled(parts, count, gain, threshold, readings, rng),
                                )
                            )
    return components, power


# A side's placements hold a row of every stay per draw: a chunk of draws at a time keeps a
# budget's thirty-odd sides within a few hundred megabytes, each side read once per chunk.
CHUNK = 250


def _read_pairs(
    answers: Mapping[SideKey, Answers],
    pairs: Sequence[Pair],
    counts: Counts,
    outcome: NDArray[np.bool_],
) -> dict[Pair, tuple[Floats, Floats]]:
    """Each pair's per-seed differences and pooled difference's interval error, on every draw."""
    sides = sorted({side for p in pairs for side in (p.control, p.candidate)})
    differences: dict[Pair, list[Floats]] = {p: [] for p in pairs}
    errors: dict[Pair, list[Floats]] = {p: [] for p in pairs}
    for start in range(0, counts.shape[0], CHUNK):
        chunk = counts[start : start + CHUNK]
        read = {side: drawn_side(answers[side], chunk) for side in sides}
        for p in pairs:
            control, candidate = read[p.control], read[p.candidate]
            differences[p].append(candidate.areas - control.areas)
            errors[p].append(pooled_error(control, candidate, chunk, outcome))
    return {p: (np.concatenate(differences[p]), np.concatenate(errors[p])) for p in pairs}


def population_gap(control: Answers, candidate: Answers) -> float:
    """The difference in area pooled over seeds, on every exported stay."""
    every = np.ones((1, len(control.outcome)), dtype=np.int64)
    return float(
        drawn_side(candidate, every).areas.mean() - drawn_side(control, every).areas.mean()
    )


class CheckRow(NamedTuple):
    control: str
    candidate: str
    budget: str
    stays: int
    bootstrap_error: float
    placement_error: float

    @property
    def ratio(self) -> float:
        return self.placement_error / self.bootstrap_error


def check(predictions: Path, comparisons: Path) -> list[CheckRow]:
    """Placement errors against the bootstrap intervals of pairs read by area."""
    with comparisons.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["measure"] == "auroc_shortfall"]
    campaigns = {r["control_campaign"] for r in rows} | {r["candidate_campaign"] for r in rows}
    answers = read_answers([predictions], campaigns)
    found = []
    for r in rows:
        control = answers[SideKey(r["control_campaign"], r["control"], r["budget"])]
        candidate = answers[SideKey(r["candidate_campaign"], r["candidate"], r["budget"])]
        every = np.ones((1, len(control.outcome)), dtype=np.int64)
        error = pooled_error(
            drawn_side(control, every), drawn_side(candidate, every), every, control.outcome
        )
        found.append(
            CheckRow(
                control=f"{r['control_campaign'][:8]} {r['control']}",
                candidate=f"{r['candidate_campaign'][:8]} {r['candidate']}",
                budget=r["budget"],
                stays=len(control.outcome),
                bootstrap_error=(float(r["high"]) - float(r["low"])) / (2 * Z_95),
                placement_error=float(error[0]),
            )
        )
    return found


def write(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(("" if v is None else v for v in row) for row in rows)


def median_power(
    power: Sequence[PowerRow],
    kind: str,
    budget: str,
    stays: int,
    seeds: int,
    gain: float,
    source: str,
    threshold: float | None,
) -> tuple[float, float, float] | None:
    """The median over a kind's pairs, and the pairs' range."""
    values = [
        r.power
        for r in power
        if (r.kind, r.budget, r.stays, r.seeds, r.gain, r.source, r.threshold)
        == (kind, budget, stays, seeds, gain, source, threshold)
    ]
    if not values:
        return None
    return statistics.median(values), min(values), max(values)


def render(
    components: Sequence[ComponentRow],
    power: Sequence[PowerRow],
    checked: Sequence[CheckRow],
    design: Design = DEFAULT_DESIGN,
) -> str:
    sections: list[str] = []
    if checked:
        ratios = [c.ratio for c in checked]
        sections.append(
            f"Placement error against bootstrap: {len(checked)} pairs, ratio median "
            f"{statistics.median(ratios):.3f}, range {min(ratios):.3f} to {max(ratios):.3f}"
        )
    sections.append(table(*_components_table(components)))
    budgets = sorted({r.budget for r in power}, key=budget_order)
    kinds = sorted({r.kind for r in power})
    for budget in budgets:
        for kind in kinds:
            for threshold in design.thresholds_of(kind, budget):
                rows = list(_power_rows(power, kind, budget, threshold, design))
                if rows:
                    note = "" if threshold is None else f", spread between backbones {threshold}"
                    sections.append(
                        f"Kind {kind} at {budget} stays{note}: median power over pairs [range]"
                    )
                    sections.append(
                        table(("stays", "source", "seeds", *(f"δ={g}" for g in design.gains)), rows)
                    )
    return "\n\n".join(sections)


def _components_table(
    components: Sequence[ComponentRow],
) -> tuple[tuple[str, ...], Iterator[tuple[str, ...]]]:
    header = (
        "kind",
        "budget",
        "stays",
        "pairs",
        "seed",
        "stays part",
        "interaction",
        "interval SE",
        "seed SE",
    )
    keys = sorted({(c.kind, c.budget, c.stays) for c in components}, key=_order)

    def rows() -> Iterator[tuple[str, ...]]:
        for kind, budget, stays in keys:
            group = [c for c in components if (c.kind, c.budget, c.stays) == (kind, budget, stays)]
            yield (
                kind,
                budget,
                str(stays),
                str(len(group)),
                f"{statistics.median(c.seed for c in group):.4f}",
                f"{statistics.median(c.stays_part for c in group):.4f}",
                f"{statistics.median(c.interaction for c in group):.4f}",
                f"{statistics.median(c.mean_interval_error for c in group):.4f}",
                f"{statistics.median(c.seed_error for c in group):.4f}",
            )

    return header, rows()


def _order(key: tuple[str, str, int]) -> tuple[str, float, int]:
    kind, budget, stays = key
    return kind, budget_order(budget), stays


def _power_rows(
    power: Sequence[PowerRow], kind: str, budget: str, threshold: float | None, design: Design
) -> Iterator[tuple[str, ...]]:
    counts = sorted({r.stays for r in power if (r.kind, r.budget) == (kind, budget)})
    ran = {r.seeds for r in power if r.source == "direct"}
    for stays in counts:
        designs = [
            *(("direct", k) for k in sorted(ran)),
            *(("measured seeds", k) for k in sorted(ran)),
            *(("model", k) for k in design.seeds),
        ]
        for source, seeds in designs:
            cells = [
                median_power(power, kind, budget, stays, seeds, g, source, threshold)
                for g in design.gains
            ]
            if all(c is None for c in cells):
                continue
            yield (
                str(stays),
                source,
                str(seeds),
                *("—" if c is None else f"{c[0]:.2f} [{c[1]:.2f}; {c[2]:.2f}]" for c in cells),
            )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True, metavar="CSV")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument(
        "--pair",
        nargs=6,
        action="append",
        default=[],
        metavar=(
            "KIND",
            "LABEL",
            "CONTROL_CAMPAIGN",
            "CONTROL",
            "CANDIDATE_CAMPAIGN",
            "CANDIDATE",
        ),
    )
    parser.add_argument("--budgets", nargs="+", default=[], metavar="BUDGET")
    parser.add_argument("--stays", nargs="+", type=int, default=[], metavar="N")
    parser.add_argument("--gains", nargs="+", type=float, default=list(GAINS), metavar="GAIN")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), metavar="K")
    parser.add_argument(
        "--seed-gap", nargs=2, action="append", default=[], metavar=("KIND", "VALUE")
    )
    parser.add_argument("--check-predictions", type=Path, metavar="CSV")
    parser.add_argument("--check-comparisons", type=Path, metavar="CSV")
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--readings", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    if (arguments.check_predictions is None) != (arguments.check_comparisons is None):
        parser.error("--check-predictions and --check-comparisons go together")
    if not arguments.pair:
        parser.error("name at least one --pair")
    return arguments


def design_of(arguments: argparse.Namespace) -> Design:
    gaps: dict[str, tuple[tuple[str | None, float], ...]] = {}
    for kind, value in arguments.seed_gap:
        at, _, spread = value.rpartition("=")
        gaps[kind] = (*gaps.get(kind, ()), (at or None, float(spread)))
    return Design(
        stays=tuple(arguments.stays),
        seeds=tuple(arguments.seeds),
        gains=tuple(arguments.gains),
        seed_gaps=gaps,
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    checked: list[CheckRow] = []
    if arguments.check_predictions is not None:
        checked = check(arguments.check_predictions, arguments.check_comparisons)
        write(
            out / CHECK,
            (*CheckRow._fields, "ratio"),
            ((*c, c.ratio) for c in checked),
        )
    specs = [PairSpec(*each) for each in arguments.pair]
    campaigns = {s.control_campaign for s in specs} | {s.candidate_campaign for s in specs}
    answers = read_answers(arguments.predictions, campaigns)
    pairs = pairs_of(specs, answers, arguments.budgets)
    design = design_of(arguments)
    components, power = simulate(
        answers,
        pairs,
        draws=arguments.draws,
        readings=arguments.readings,
        seed=arguments.seed,
        design=design,
    )
    write(out / COMPONENTS, ComponentRow._fields, components)
    write(out / POWER, PowerRow._fields, power)
    print(render(components, power, checked, design))


if __name__ == "__main__":
    main()
