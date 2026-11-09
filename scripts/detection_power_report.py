"""How often a detection campaign confirms a true gain in event-weighted area, from its events.

A detection campaign compares two detectors on the validation windows of one task, paired by
unit and resampled by unit in two strata, units holding an event's window and units holding
none; the endpoint is confirmed when the reduction in shortfall reaches the registered least
gain and the paired interval lies above zero. How often that happens at a true gain depends on
how many events the side holds, how many windows each spans and how the events are spread over
the units. This script reads that structure off the raw annotations, as the harness will read
it, and simulates the rule over scores of a stated model.

The model: every normal window scores a standard normal draw; every window of an event scores
that draw plus a shift shared by the event's windows, drawn per event around a mean, so events
differ in how far they stand out and a long event's windows move together. The control's mean
shift is set so that its expected area is the level stated; the candidate adds a shift to the
events' windows alone, set so that its expected area is the level plus the gain. Both sides
share the noise, so the simulated difference carries the sampling of units and events and
nothing of a detector's own seed, which a deterministic detector has none of. The power is the
share of simulated sides the rule confirms.

    uv run scripts/detection_power_report.py structure --task esa-mission1-events --out DIR
    uv run scripts/detection_power_report.py simulate --structure DIR/structure.csv --out DIR
        [--level 0.3 ...] [--gains 0.02 0.05 0.1] [--minimum 0.02] [--worlds 200]
        [--resamples 1000]
    uv run scripts/detection_power_report.py interaction --structure DIR/structure.csv --out DIR
        [--level 0.1 ...] [--teaching 0.05] [--gains 0.02 0.05 0.1] [--minimum 0.02]
        [--worlds 200] [--resamples 1000]
    uv run scripts/detection_power_report.py recall --structure DIR/structure.csv --out DIR
        [--share 0.01] [--level 0.4 0.6] [--gains 0.05 0.1] [--minimum 0.016]
        [--models shared jittered onset] [--jitter 1.5] [--worlds 200] [--resamples 1000]
    uv run scripts/detection_power_report.py jitter --structure DIR/structure.csv --out DIR
        [--share 0.01] [--level 0.6] [--jitters 0 1 2] [--worlds 50] [--resamples 500]
    uv run scripts/detection_power_report.py injected --predictions FILE --pairs FILE --out DIR
        [--share 0.01] [--gains 0.12 0.16 0.2] [--minimum 0.016] [--worlds 100] [--resamples 1000]

``structure`` writes one row per validation window of the task, with its unit and the event it
overlaps, from the raw files under ``data/raw``; ``simulate`` writes ``power.csv`` and prints
the table a note pastes. ``interaction`` simulates a grid of two detectors by two teachings:
both sides lift by the teaching gain when taught, the candidate by the interaction gain more,
and the rule confirms the difference of the two paired reductions, every side on one noise and
one draw of the units; it writes ``interaction-power.csv``. The areas are computed by the same
sweep the harness reads, checked against it on a small case.

``recall`` asks the same of the events found at a share of normal windows flagged. A recall rests
on each event's best window, so the area's model flatters it: a candidate that only adds to every
event never loses one, and the interval comes out narrow. Two more models answer that. ``jittered``
adds a draw per event to the candidate's lift, so some events are lost while others are found, its
size set by ``jitter``, which reads the interval's half-width at no gain against an observed one.
``onset`` also puts the shift on each event's first window alone, as a detector referenced to its
own history does. ``injected`` leaves the models behind: it reads real pairs of detectors from an
export of a campaign's predictions and raises the candidate's best window on randomly chosen
events it misses, to a score its found events reach, until the pair's gain is the one asked for.
The two detectors' real disagreement and the real clustering of events in months stay, and only
the power and the interval's width are printed.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS, EsaAdRawCorpusReader
from emblema.entrypoints.cli.campaign.known_tasks import HeldOutShare, KnownTask, KnownTasks
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
from emblema.evaluation.adapters.readers.opssat_ad_ground_truth import OpssatAdGroundTruth
from emblema.evaluation.adapters.readers.skab_ground_truth import SkabGroundTruth
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.ports.ground_truth import GroundTruth
from scripts.raw_corpora import raw_root
from scripts.reporting import table

STRUCTURE, POWER, INTERACTION_POWER = "structure.csv", "power.csv", "interaction-power.csv"
RECALL_POWER, JITTER_WIDTHS = "recall-power.csv", "jitter-widths.csv"
INJECTED_POWER = "injected-power.csv"
SHARED, JITTERED, ONSET = "shared", "jittered", "onset"
RECALL_MODELS = (SHARED, JITTERED, ONSET)
STRUCTURE_COLUMNS = ("unit", "position", "event")
Floats = NDArray[np.float64]
Ints = NDArray[np.int64]


@dataclass(frozen=True)
class Structure:
    """The validation windows of a task: which unit each belongs to and which event, if any.

    Attributes:
        units: Index of each window's unit.
        events: The event each window overlaps, zero for none.
        unit_count: How many units there are.
    """

    units: Ints
    events: Ints
    unit_count: int

    @classmethod
    def read(cls, path: Path) -> "Structure":
        rows = list(csv.DictReader(path.open(newline="")))
        names = sorted({row["unit"] for row in rows})
        index = {name: place for place, name in enumerate(names)}
        return cls(
            units=np.array([index[row["unit"]] for row in rows], dtype=np.int64),
            events=np.array([int(row["event"]) for row in rows], dtype=np.int64),
            unit_count=len(names),
        )

    @property
    def weights(self) -> Floats:
        """What each window weighs in the area: one over its event's windows, or one."""
        counts = np.bincount(self.events)
        return np.where(self.events > 0, 1.0 / counts[self.events], 1.0)

    @property
    def strata(self) -> tuple[Ints, Ints]:
        """The units holding an event's window, and the units holding none."""
        holding = np.zeros(self.unit_count, dtype=bool)
        holding[self.units[self.events > 0]] = True
        everyone = np.arange(self.unit_count)
        return everyone[holding], everyone[~holding]


def average_precision(scores: Floats, events: Ints, weights: Floats, unit_weights: Floats) -> float:
    """The event-weighted area, by the harness's sweep in numpy: ties swept as one group."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    counted = weights[order] * unit_weights[order]
    positive = events[order] > 0
    found = np.cumsum(np.where(positive, counted, 0.0))
    alarmed = np.cumsum(np.where(positive, 0.0, counted))
    closes = np.ones(len(order), dtype=bool)
    closes[:-1] = sorted_scores[1:] != sorted_scores[:-1]
    found_at_close = found[closes]
    alarmed_at_close = alarmed[closes]
    found_in_group = np.diff(found_at_close, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(
            found_at_close + alarmed_at_close > 0.0,
            found_at_close / (found_at_close + alarmed_at_close),
            0.0,
        )
    total = found_at_close[-1]
    if total == 0.0 or alarmed_at_close[-1] == 0.0:
        raise ValueError("the weights leave no event or no normal window")
    return float(np.sum(found_in_group * precision) / total)


def check_against_the_harness() -> None:
    """The sweep above agrees with the domain's on a small case, or the script stops."""
    rows = [
        ("a", 0, 1, 0.9),
        ("a", 1, 1, 0.8),
        ("a", 2, 0, 0.6),
        ("b", 0, 2, 0.5),
        ("b", 1, 0, 0.5),
    ]
    expected = AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
            target=float(event),
            predicted=score,
        )
        for unit, position, event, score in rows
    ).average_precision_weighted([2, 1])
    structure = Structure(
        units=np.array([0, 0, 0, 1, 1]), events=np.array([1, 1, 0, 2, 0]), unit_count=2
    )
    scores = np.array([0.9, 0.8, 0.6, 0.5, 0.5])
    got = average_precision(
        scores, structure.events, structure.weights, np.array([2.0, 1.0])[structure.units]
    )
    if abs(got - expected) > 1e-12:
        raise SystemExit(f"the sweep disagrees with the harness: {got} against {expected}")
    ranking = AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
            target=float(event),
            predicted=score,
        )
        for unit, position, event, score in rows
    )
    once = np.ones((1, structure.unit_count))
    for share in (0.0, 0.5, 1.0):
        found = float(event_recalls(structure, scores, once, share)[0])
        wanted = ranking.event_recall_at_false_alarm_share(share)
        if abs(found - wanted) > 1e-12:
            raise SystemExit(f"the recall disagrees with the harness: {found} against {wanted}")


def shift_for(structure: Structure, level: float, draws: np.random.Generator, worlds: int) -> float:
    """The mean shift of the events' windows at which the expected area is ``level``."""
    low, high = 0.0, 8.0
    for _ in range(30):
        middle = (low + high) / 2.0
        area = float(
            np.mean(
                [
                    average_precision(
                        scores_of(structure, middle, draws),
                        structure.events,
                        structure.weights,
                        np.ones(len(structure.units)),
                    )
                    for _ in range(worlds)
                ]
            )
        )
        if area < level:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def scores_of(structure: Structure, shift: float, draws: np.random.Generator) -> Floats:
    """One world of scores: noise per window, a shift per event around ``shift``."""
    noise = draws.standard_normal(len(structure.events))
    per_event = draws.normal(shift, shift / 2.0, size=int(structure.events.max()) + 1)
    per_event[0] = 0.0
    return noise + per_event[structure.events]


def confirms(
    structure: Structure,
    control: Floats,
    candidate: Floats,
    *,
    minimum: float,
    resamples: int,
    draws: np.random.Generator,
) -> bool:
    """Whether the harness's rule confirms the candidate over the control on this world."""
    weights = structure.weights
    ones = np.ones(len(structure.units))
    reduction = average_precision(candidate, structure.events, weights, ones) - average_precision(
        control, structure.events, weights, ones
    )
    if reduction < minimum:
        return False
    holding, quiet = structure.strata
    reductions = np.empty(resamples)
    for place in range(resamples):
        picks = np.concatenate(
            (
                draws.choice(holding, size=len(holding), replace=True),
                draws.choice(quiet, size=len(quiet), replace=True),
            )
        )
        unit_weights = np.bincount(picks, minlength=structure.unit_count).astype(np.float64)
        counted = unit_weights[structure.units]
        reductions[place] = average_precision(
            candidate, structure.events, weights, counted
        ) - average_precision(control, structure.events, weights, counted)
    return bool(np.quantile(reductions, 0.025) > 0.0)


def _unit_weights(structure: Structure, draws: np.random.Generator) -> Floats:
    """One resample of the units, each stratum on its own, as a weight per window."""
    holding, quiet = structure.strata
    picks = np.concatenate(
        (
            draws.choice(holding, size=len(holding), replace=True),
            draws.choice(quiet, size=len(quiet), replace=True),
        )
    )
    counted: Floats = np.bincount(picks, minlength=structure.unit_count).astype(np.float64)[
        structure.units
    ]
    return counted


def confirms_interaction(
    structure: Structure,
    sides: tuple[Floats, Floats, Floats, Floats],
    *,
    minimum: float,
    resamples: int,
    draws: np.random.Generator,
) -> bool:
    """Whether the rule confirms the interaction on this world.

    ``sides`` are the scores of the untaught control, the untaught candidate, the taught
    control and the taught candidate; the interaction is the taught pair's reduction less the
    untaught pair's, every resample of the units moving all four together.
    """
    weights = structure.weights

    def interaction(unit_weights: Floats) -> float:
        areas = [
            average_precision(scores, structure.events, weights, unit_weights) for scores in sides
        ]
        return (areas[3] - areas[2]) - (areas[1] - areas[0])

    if interaction(np.ones(len(structure.units))) < minimum:
        return False
    resampled = np.array([interaction(_unit_weights(structure, draws)) for _ in range(resamples)])
    return bool(np.quantile(resampled, 0.025) > 0.0)


def simulate_interaction(
    structure: Structure,
    *,
    levels: Sequence[float],
    teaching: float,
    gains: Sequence[float],
    minimum: float,
    worlds: int,
    resamples: int,
    seed: int,
) -> list[dict[str, str]]:
    """The power of the interaction's rule: both sides lifted by ``teaching``, one by more."""
    draws = np.random.default_rng(seed)
    rows = []
    for level in levels:
        base = shift_for(structure, level, draws, 50)
        lifted = shift_for(structure, min(level + teaching, 0.999), draws, 50)
        for gain in gains:
            raised = shift_for(structure, min(level + teaching + gain, 0.999), draws, 50)
            confirmed = 0
            for _ in range(worlds):
                noise = draws.standard_normal(len(structure.events))
                per_event = draws.normal(base, base / 2.0, size=int(structure.events.max()) + 1)
                per_event[0] = 0.0
                control = noise + per_event[structure.events]
                in_events = structure.events > 0
                sides = (
                    control,
                    control,
                    control + np.where(in_events, lifted - base, 0.0),
                    control + np.where(in_events, raised - base, 0.0),
                )
                if confirms_interaction(
                    structure, sides, minimum=minimum, resamples=resamples, draws=draws
                ):
                    confirmed += 1
            rows.append(
                {
                    "level": f"{level:g}",
                    "teaching": f"{teaching:g}",
                    "gain": f"{gain:g}",
                    "minimum": f"{minimum:g}",
                    "worlds": str(worlds),
                    "power": f"{confirmed / worlds:.2f}",
                }
            )
            print(
                f"level {level:g} teaching {teaching:g} interaction {gain:g}: "
                f"power {confirmed / worlds:.2f}",
                flush=True,
            )
    return rows


def simulate(
    structure: Structure,
    *,
    levels: Sequence[float],
    gains: Sequence[float],
    minimum: float,
    worlds: int,
    resamples: int,
    seed: int,
) -> list[dict[str, str]]:
    draws = np.random.default_rng(seed)
    rows = []
    for level in levels:
        base = shift_for(structure, level, draws, 50)
        for gain in gains:
            raised = shift_for(structure, min(level + gain, 0.999), draws, 50)
            confirmed = 0
            for _ in range(worlds):
                noise = draws.standard_normal(len(structure.events))
                per_event = draws.normal(base, base / 2.0, size=int(structure.events.max()) + 1)
                per_event[0] = 0.0
                control = noise + per_event[structure.events]
                candidate = control + np.where(structure.events > 0, raised - base, 0.0)
                if confirms(
                    structure,
                    control,
                    candidate,
                    minimum=minimum,
                    resamples=resamples,
                    draws=draws,
                ):
                    confirmed += 1
            rows.append(
                {
                    "level": f"{level:g}",
                    "gain": f"{gain:g}",
                    "minimum": f"{minimum:g}",
                    "worlds": str(worlds),
                    "power": f"{confirmed / worlds:.2f}",
                }
            )
            print(f"level {level:g} gain {gain:g}: power {confirmed / worlds:.2f}", flush=True)
    return rows


def event_recalls(structure: Structure, scores: Floats, counts: Floats, share: float) -> Floats:
    """The share of events found per draw of units, by the harness's rule in numpy.

    ``counts`` is ``[draws, units]``. A window of an event is flagged when the normal windows
    scoring at least as high, each counted as often as its unit, stay within ``share`` of the
    normal windows; an event weighs what its windows weigh in the draw.
    """
    hits = np.flatnonzero(structure.events > 0)
    order = hits[np.argsort(structure.events[hits], kind="stable")]
    hit_units = structure.units[order]
    normal = structure.events == 0
    normal_units = structure.units[normal]
    normal_scores = scores[normal]
    above = np.zeros((structure.unit_count, len(order)), dtype=np.float64)
    for unit in range(structure.unit_count):
        sorted_scores = np.sort(normal_scores[normal_units == unit])
        above[unit] = len(sorted_scores) - np.searchsorted(sorted_scores, scores[order], "left")
    normals_per_unit = np.bincount(normal_units, minlength=structure.unit_count)
    total = counts @ normals_per_unit.astype(np.float64)
    if not total.all():
        raise ValueError("the draw leaves no normal window")
    held = counts[:, hit_units]
    flagged = (counts @ above <= share * total[:, None]) & (held > 0.0)
    weighed = held * structure.weights[order]
    starts = np.flatnonzero(np.diff(structure.events[order], prepend=-1))
    found_any = np.logical_or.reduceat(flagged, starts, axis=1)
    worth = np.add.reduceat(weighed, starts, axis=1)
    weights = worth.sum(axis=1)
    if not weights.all():
        raise ValueError("the draw leaves no event")
    recalls: Floats = (worth * found_any).sum(axis=1) / weights
    return recalls


def signal_windows(structure: Structure, model: str) -> NDArray[np.bool_]:
    """The windows that carry an event's shift: all of them, or each event's first."""
    if model != ONSET:
        carrying: NDArray[np.bool_] = structure.events > 0
        return carrying
    _, first = np.unique(structure.events, return_index=True)
    onset = np.zeros(len(structure.events), dtype=bool)
    onset[first] = True
    onset &= structure.events > 0
    return onset


def draw_sides(
    structure: Structure,
    signal: NDArray[np.bool_],
    *,
    base: float,
    raised: float,
    jitter: float,
    draws: np.random.Generator,
) -> tuple[Floats, Floats]:
    """One world of a pair on one noise: the candidate lifts each event, jittered per event."""
    size = int(structure.events.max()) + 1
    per_event = draws.normal(base, base / 2.0, size=size)
    per_event[0] = 0.0
    control = draws.standard_normal(len(structure.events)) + signal * per_event[structure.events]
    lift = draws.normal(raised - base, jitter, size=size)
    lift[0] = 0.0
    return control, control + signal * lift[structure.events]


def recall_shift(
    structure: Structure,
    signal: NDArray[np.bool_],
    level: float,
    *,
    share: float,
    worlds: int,
    draws: np.random.Generator,
    base: float | None = None,
    jitter: float = 0.0,
) -> float:
    """The shift at which the expected recall is ``level``: the control's, or above ``base``."""
    once = np.ones((1, structure.unit_count))
    # A candidate's jitter alone lifts its recall, so its mean lift may have to be negative.
    low, high = (0.0 if base is None else -12.0), 12.0
    for _ in range(24):
        middle = (low + high) / 2.0
        reached = 0.0
        for _ in range(worlds):
            sides = draw_sides(
                structure,
                signal,
                base=middle if base is None else base,
                raised=middle,
                jitter=jitter,
                draws=draws,
            )
            scored = sides[0] if base is None else sides[1]
            reached += float(event_recalls(structure, scored, once, share)[0])
        if reached / worlds < level:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def resampled_counts(structure: Structure, resamples: int, draws: np.random.Generator) -> Floats:
    """``[resamples, units]``: each unit's count in a draw of the units, stratum by stratum."""
    holding, quiet = structure.strata
    counts = np.zeros((resamples, structure.unit_count), dtype=np.float64)
    for place in range(resamples):
        picks = np.concatenate(
            (
                draws.choice(holding, size=len(holding), replace=True),
                draws.choice(quiet, size=len(quiet), replace=True),
            )
        )
        counts[place] = np.bincount(picks, minlength=structure.unit_count)
    return counts


def recall_interval(
    structure: Structure,
    control: Floats,
    candidate: Floats,
    *,
    share: float,
    resamples: int,
    draws: np.random.Generator,
) -> tuple[float, float, float]:
    """The candidate's gain in recall and the 95 % interval over draws of the units."""
    once = np.ones((1, structure.unit_count))
    gain = float(
        event_recalls(structure, candidate, once, share)[0]
        - event_recalls(structure, control, once, share)[0]
    )
    counts = resampled_counts(structure, resamples, draws)
    gains = event_recalls(structure, candidate, counts, share) - event_recalls(
        structure, control, counts, share
    )
    return gain, float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def confirms_recall(gain: float, lower: float, *, minimum: float) -> bool:
    """Whether the rule confirms a gain: past the floor, and an interval above zero."""
    return gain >= minimum and lower > 0.0


def simulate_recall(
    structure: Structure,
    *,
    share: float,
    levels: Sequence[float],
    gains: Sequence[float],
    models: Sequence[str],
    jitter: float,
    minimum: float,
    worlds: int,
    resamples: int,
    seed: int,
    shift_worlds: int = 30,
) -> list[dict[str, str]]:
    """The power of the recall rule per level, true gain and model of a detector."""
    draws = np.random.default_rng(seed)
    rows = []
    for model in models:
        signal = signal_windows(structure, model)
        spread = 0.0 if model == SHARED else jitter
        for level in levels:
            base = recall_shift(
                structure, signal, level, share=share, worlds=shift_worlds, draws=draws
            )
            for gain in gains:
                raised = recall_shift(
                    structure,
                    signal,
                    min(level + gain, 0.999),
                    share=share,
                    worlds=shift_worlds,
                    draws=draws,
                    base=base,
                    jitter=spread,
                )
                confirmed = 0
                widths = []
                realised = []
                for _ in range(worlds):
                    control, candidate = draw_sides(
                        structure, signal, base=base, raised=raised, jitter=spread, draws=draws
                    )
                    observed, lower, upper = recall_interval(
                        structure,
                        control,
                        candidate,
                        share=share,
                        resamples=resamples,
                        draws=draws,
                    )
                    confirmed += confirms_recall(observed, lower, minimum=minimum)
                    widths.append((upper - lower) / 2.0)
                    realised.append(observed)
                rows.append(
                    {
                        "model": model,
                        "jitter": f"{spread:g}",
                        "level": f"{level:g}",
                        "gain": f"{gain:g}",
                        "minimum": f"{minimum:g}",
                        "worlds": str(worlds),
                        "power": f"{confirmed / worlds:.2f}",
                        "half_width": f"{float(np.median(widths)):.3f}",
                        "realised_gain": f"{float(np.mean(realised)):.3f}",
                    }
                )
                print(
                    f"{model} level {level:g} gain {gain:g}: power {confirmed / worlds:.2f}, "
                    f"half-width {float(np.median(widths)):.3f}",
                    flush=True,
                )
    return rows


def jitter_widths(
    structure: Structure,
    *,
    share: float,
    level: float,
    jitters: Sequence[float],
    worlds: int,
    resamples: int,
    seed: int,
    shift_worlds: int = 30,
) -> list[dict[str, str]]:
    """The interval's half-width at no true gain, per jitter and model of the signal."""
    draws = np.random.default_rng(seed)
    rows = []
    for model in (JITTERED, ONSET):
        signal = signal_windows(structure, model)
        base = recall_shift(structure, signal, level, share=share, worlds=shift_worlds, draws=draws)
        for jitter in jitters:
            raised = recall_shift(
                structure,
                signal,
                level,
                share=share,
                worlds=shift_worlds,
                draws=draws,
                base=base,
                jitter=jitter,
            )
            widths = []
            for _ in range(worlds):
                control, candidate = draw_sides(
                    structure, signal, base=base, raised=raised, jitter=jitter, draws=draws
                )
                _, lower, upper = recall_interval(
                    structure, control, candidate, share=share, resamples=resamples, draws=draws
                )
                widths.append((upper - lower) / 2.0)
            width = float(np.median(widths))
            rows.append(
                {
                    "model": model,
                    "level": f"{level:g}",
                    "jitter": f"{jitter:g}",
                    "half_width": f"{width:.3f}",
                }
            )
            print(f"{model} jitter {jitter:g}: half-width {width:.3f}", flush=True)
    return rows


def found_events(structure: Structure, scores: Floats, share: float) -> Ints:
    """The events with a window flagged at ``share`` of the normal windows, sorted."""
    normal = np.sort(scores[structure.events == 0])
    above = len(normal) - np.searchsorted(normal, scores, "left")
    flagged = (above <= share * len(normal)) & (structure.events > 0)
    return np.unique(structure.events[flagged])


def best_windows(structure: Structure, scores: Floats, events: Ints) -> Ints:
    """Each event's highest-scoring window, the earliest among equals."""
    best = np.empty(len(events), dtype=np.int64)
    for place, event in enumerate(events.tolist()):
        members = np.flatnonzero(structure.events == event)
        best[place] = members[int(np.argmax(scores[members]))]
    return best


def injected_candidate(
    structure: Structure,
    control: Floats,
    candidate: Floats,
    *,
    gain: float,
    share: float,
    draws: np.random.Generator,
) -> Floats:
    """The candidate with enough missed events found that its gain over the control is ``gain``.

    Each chosen event's best window takes the score of a randomly chosen found event's best
    window, so an injected event stands as far above the alarm as the real ones do.

    Raises:
        ValueError: If the candidate misses too few events, or finds none to copy a score from.
    """
    every = np.unique(structure.events[structure.events > 0])
    found = found_events(structure, candidate, share)
    wanted = round(gain * len(every)) - (len(found) - len(found_events(structure, control, share)))
    if wanted <= 0:
        return candidate.copy()
    missed = np.setdiff1d(every, found)
    if wanted > len(missed) or not len(found):
        raise ValueError(f"cannot find {wanted} more events: {len(missed)} are missed")
    chosen = draws.choice(missed, size=wanted, replace=False)
    donors = candidate[best_windows(structure, candidate, found)]
    raised = candidate.copy()
    raised[best_windows(structure, candidate, chosen)] = draws.choice(donors, size=wanted)
    return raised


def read_pairs(predictions: Path, plan: Path) -> tuple[Structure, list[tuple[str, Floats, Floats]]]:
    """The windows of each planned pair, aligned, from an export of predictions.

    ``plan`` holds one pair a line, tab-separated: control campaign, control candidate, candidate
    campaign, candidate candidate. A detector's first seed is read.

    Raises:
        SystemExit: If a detector is missing or the pairs do not score the same windows.
    """
    pairs = [tuple(line.split("\t")) for line in plan.read_text().splitlines() if line.strip()]
    wanted = {key for pair in pairs for key in (pair[:2], pair[2:])}
    scores: dict[tuple[str, ...], dict[tuple[str, int], float]] = {key: {} for key in wanted}
    targets: dict[tuple[str, int], int] = {}
    first_seed: dict[tuple[str, ...], str] = {}
    with predictions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["campaign"], row["candidate"])
            if key not in wanted or first_seed.setdefault(key, row["seed"]) != row["seed"]:
                continue
            window = (row["unit"], int(row["position"]))
            scores[key][window] = float(row["predicted"])
            targets[window] = int(float(row["target"]))
    windows: set[tuple[str, int]] = set(targets)
    for detector, held in scores.items():
        if set(held) != windows:
            raise SystemExit(
                f"{detector} scores {len(held)} windows of the {len(windows)} asked for"
            )
    ordered = sorted(windows)
    names = sorted({unit for unit, _ in ordered})
    index = {name: place for place, name in enumerate(names)}
    structure = Structure(
        units=np.array([index[unit] for unit, _ in ordered], dtype=np.int64),
        events=np.array([targets[window] for window in ordered], dtype=np.int64),
        unit_count=len(names),
    )

    def array(key: tuple[str, ...]) -> Floats:
        return np.array([scores[key][window] for window in ordered], dtype=np.float64)

    return structure, [
        (f"{pair[0][:8]}:{pair[1]} / {pair[2][:8]}:{pair[3]}", array(pair[:2]), array(pair[2:]))
        for pair in pairs
    ]


def simulate_injected(
    structure: Structure,
    pairs: Sequence[tuple[str, Floats, Floats]],
    *,
    share: float,
    gains: Sequence[float],
    minimum: float,
    worlds: int,
    resamples: int,
    seed: int,
) -> list[dict[str, str]]:
    """The recall rule's power on real pairs, per injected gain, pooled and across pairs."""
    draws = np.random.default_rng(seed)
    rows = []
    for gain in gains:
        powers = []
        widths = []
        for _, control, candidate in pairs:
            confirmed = 0
            for _ in range(worlds):
                raised = injected_candidate(
                    structure, control, candidate, gain=gain, share=share, draws=draws
                )
                observed, lower, upper = recall_interval(
                    structure, control, raised, share=share, resamples=resamples, draws=draws
                )
                confirmed += confirms_recall(observed, lower, minimum=minimum)
                widths.append((upper - lower) / 2.0)
            powers.append(confirmed / worlds)
        rows.append(
            {
                "gain": f"{gain:g}",
                "pairs": str(len(pairs)),
                "worlds_per_pair": str(worlds),
                "minimum": f"{minimum:g}",
                "power": f"{float(np.mean(powers)):.3f}",
                "power_low": f"{min(powers):.2f}",
                "power_high": f"{max(powers):.2f}",
                "half_width": f"{float(np.median(widths)):.3f}",
            }
        )
        print(
            f"gain {gain:g}: power {float(np.mean(powers)):.3f} "
            f"[{min(powers):.2f}; {max(powers):.2f}]",
            flush=True,
        )
    return rows


def structure_of(task_name: str, corpora: Path) -> list[tuple[str, int, int]]:
    """Every validation window of the task, with its unit and event, from the raw files."""
    task = KnownTasks.named(task_name)
    if task.corpus == KnownGroundTruths.ESA_AD_RAW:
        mission = task.unit_prefix.rstrip("/")
        root = raw_root("esa_ad")
        if root is None:
            raise SystemExit("the ESA Anomaly Dataset is not downloaded")
        truth: GroundTruth = EsaAdRawGroundTruth(
            root,
            channels=EsaAdRawCorpusReader.POLICY.channels,
            origin_seconds=ORIGIN_SECONDS,
            window_hours=KnownGroundTruths.ESA_AD_RAW_WINDOW_HOURS,
        )
        first, last = {
            "ESA-Mission1": ("2007-01", "2013-12"),
            "ESA-Mission2": ("2001-10", "2003-06"),
        }[mission]
        held = months_between(mission, first, last)
        windows = [
            TaskWindow(unit=UnitKey(key), position=position, ends_at=hour)
            for key in _validation(task, held)
            for position, hour in enumerate(window_ends_of(key))
        ]
        truths = truth.truths_of(task.corpus, windows)
        return [
            (str(w.unit), w.position, int(truths[w]))
            for w in windows
            if truths[w] != AnomalyScheme.EXCLUDED
        ]
    if task.corpus == KnownGroundTruths.SKAB:
        root = raw_root("skab")
        if root is None:
            raise SystemExit("SKAB is not downloaded")
        truth = SkabGroundTruth(root, window_seconds=KnownGroundTruths.SKAB_WINDOW_SECONDS)
        experiments = sorted(
            f"{path.parent.name}/{path.stem}"
            for folder in ("other", "valve1", "valve2")
            for path in (root / folder).glob("*.csv")
        )
        windows = []
        for key in _validation(task, experiments):
            folder, _, name = key.partition("/")
            rows = (root / folder / f"{name}.csv").read_text().splitlines()[1:]
            seconds = len([row for row in rows if row.strip()])
            ends = np.arange(100.0, seconds + 1.0, 10.0)
            windows.extend(
                TaskWindow(unit=UnitKey(key), position=position, ends_at=float(end))
                for position, end in enumerate(ends)
            )
        truths = truth.truths_of(task.corpus, windows)
        return [(str(w.unit), w.position, int(truths[w])) for w in windows]
    if task.corpus == KnownGroundTruths.OPSSAT_AD:
        root = raw_root("opssat_ad")
        if root is None:
            raise SystemExit("OPSSAT-AD is not downloaded")
        truth = OpssatAdGroundTruth(root)
        segments = sorted(
            {
                f"test/{row['segment']}"
                for row in csv.DictReader((root / "segments.csv").open(newline=""))
                if row["train"] == "0"
            },
            key=lambda key: int(key.partition("/")[2]),
        )
        windows = [
            TaskWindow(unit=UnitKey(key), position=position, ends_at=1340.0)
            for position, key in enumerate(_validation(task, segments))
        ]
        truths = truth.truths_of(task.corpus, windows)
        return [(str(w.unit), w.position, int(truths[w])) for w in windows]
    raise SystemExit(f"{task_name} is not a detection task this script reads")


def _validation(task: KnownTask, held: Sequence[str]) -> list[str]:
    """The held-out units the task validates on: those its frozen share leaves."""
    if not isinstance(task.test, HeldOutShare):
        raise SystemExit(f"{task.name} freezes named units, not a share of the held-out ones")
    frozen = {str(unit) for unit in task.frozen_test(frozenset(UnitKey(key) for key in held)).units}
    return [key for key in held if key not in frozen]


def months_between(mission: str, first: str, last: str) -> list[str]:
    year, month = (int(part) for part in first.split("-"))
    last_year, last_month = (int(part) for part in last.split("-"))
    keys = []
    while (year, month) <= (last_year, last_month):
        keys.append(f"{mission}/{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return keys


def window_ends_of(key: str) -> list[float]:
    """The ends of the hourly windows of a month, in hours since the raw reader's origin."""
    _, _, stamp = key.partition("/")
    year, month = (int(part) for part in stamp.split("-"))
    start = np.datetime64(f"{year:04d}-{month:02d}-01", "s")
    end = np.datetime64(f"{year + (month == 12):04d}-{(month % 12) + 1:02d}-01", "s")
    origin = np.datetime64(ORIGIN_SECONDS, "s")
    first_hour = float((start - origin) / np.timedelta64(1, "h"))
    last_hour = float((end - origin) / np.timedelta64(1, "h"))
    hours: list[float] = np.arange(first_hour + 1.0, last_hour + 1e-9, 1.0).tolist()
    return hours


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = parser.add_subparsers(dest="what", required=True)
    structure = what.add_parser("structure")
    structure.add_argument("--task", required=True)
    structure.add_argument("--out", type=Path, required=True)
    simulate_ = what.add_parser("simulate")
    simulate_.add_argument("--structure", type=Path, required=True)
    simulate_.add_argument("--out", type=Path, required=True)
    simulate_.add_argument("--level", type=float, nargs="+", default=[0.1, 0.3])
    simulate_.add_argument("--gains", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    simulate_.add_argument("--minimum", type=float, default=0.02)
    simulate_.add_argument("--worlds", type=int, default=200)
    simulate_.add_argument("--resamples", type=int, default=1000)
    simulate_.add_argument("--seed", type=int, default=1)
    interaction_ = what.add_parser("interaction")
    interaction_.add_argument("--structure", type=Path, required=True)
    interaction_.add_argument("--out", type=Path, required=True)
    interaction_.add_argument("--level", type=float, nargs="+", default=[0.1, 0.3])
    interaction_.add_argument("--teaching", type=float, default=0.05)
    interaction_.add_argument("--gains", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    interaction_.add_argument("--minimum", type=float, default=0.02)
    interaction_.add_argument("--worlds", type=int, default=200)
    interaction_.add_argument("--resamples", type=int, default=1000)
    interaction_.add_argument("--seed", type=int, default=1)
    recall_ = what.add_parser("recall")
    recall_.add_argument("--structure", type=Path, required=True)
    recall_.add_argument("--out", type=Path, required=True)
    recall_.add_argument("--share", type=float, default=0.01)
    recall_.add_argument("--level", type=float, nargs="+", default=[0.4, 0.6])
    recall_.add_argument("--gains", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    recall_.add_argument("--minimum", type=float, default=0.016)
    recall_.add_argument("--models", nargs="+", choices=RECALL_MODELS, default=list(RECALL_MODELS))
    recall_.add_argument("--jitter", type=float, default=1.5)
    recall_.add_argument("--worlds", type=int, default=200)
    recall_.add_argument("--resamples", type=int, default=1000)
    recall_.add_argument("--seed", type=int, default=1)
    recall_.add_argument("--shift-worlds", type=int, default=30)
    jitter_ = what.add_parser("jitter")
    jitter_.add_argument("--structure", type=Path, required=True)
    jitter_.add_argument("--out", type=Path, required=True)
    jitter_.add_argument("--share", type=float, default=0.01)
    jitter_.add_argument("--level", type=float, default=0.6)
    jitter_.add_argument("--jitters", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    jitter_.add_argument("--worlds", type=int, default=50)
    jitter_.add_argument("--resamples", type=int, default=500)
    jitter_.add_argument("--seed", type=int, default=1)
    jitter_.add_argument("--shift-worlds", type=int, default=30)
    injected_ = what.add_parser("injected")
    injected_.add_argument("--predictions", type=Path, required=True)
    injected_.add_argument("--pairs", type=Path, required=True)
    injected_.add_argument("--out", type=Path, required=True)
    injected_.add_argument("--share", type=float, default=0.01)
    injected_.add_argument("--gains", type=float, nargs="+", default=[0.12, 0.16, 0.2])
    injected_.add_argument("--minimum", type=float, default=0.016)
    injected_.add_argument("--worlds", type=int, default=100)
    injected_.add_argument("--resamples", type=int, default=1000)
    injected_.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.what == "structure":
        windows = structure_of(arguments.task, REPO_ROOT / "data" / "raw")
        with (arguments.out / STRUCTURE).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(STRUCTURE_COLUMNS)
            writer.writerows(windows)
        events = {event for _, _, event in windows if event}
        print(
            f"{len(windows)} validation windows over {len({unit for unit, _, _ in windows})} "
            f"units, {len(events)} events, {sum(1 for _, _, e in windows if e)} positive windows"
        )
        return
    check_against_the_harness()
    if arguments.what == "injected":
        real, pairs = read_pairs(arguments.predictions, arguments.pairs)
        print(f"{len(pairs)} pairs over {len(real.events)} windows, {real.unit_count} units")
        rows = simulate_injected(
            real,
            pairs,
            share=arguments.share,
            gains=arguments.gains,
            minimum=arguments.minimum,
            worlds=arguments.worlds,
            resamples=arguments.resamples,
            seed=arguments.seed,
        )
        with (arguments.out / INJECTED_POWER).open("w", newline="") as handle:
            written = csv.DictWriter(handle, fieldnames=list(rows[0]))
            written.writeheader()
            written.writerows(rows)
        print(table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
        return
    read = Structure.read(arguments.structure)
    if arguments.what in ("recall", "jitter"):
        if arguments.what == "recall":
            rows = simulate_recall(
                read,
                share=arguments.share,
                levels=arguments.level,
                gains=arguments.gains,
                models=arguments.models,
                jitter=arguments.jitter,
                minimum=arguments.minimum,
                worlds=arguments.worlds,
                resamples=arguments.resamples,
                seed=arguments.seed,
                shift_worlds=arguments.shift_worlds,
            )
            written_to = arguments.out / RECALL_POWER
        else:
            rows = jitter_widths(
                read,
                share=arguments.share,
                level=arguments.level,
                jitters=arguments.jitters,
                worlds=arguments.worlds,
                resamples=arguments.resamples,
                seed=arguments.seed,
                shift_worlds=arguments.shift_worlds,
            )
            written_to = arguments.out / JITTER_WIDTHS
        with written_to.open("w", newline="") as handle:
            written = csv.DictWriter(handle, fieldnames=list(rows[0]))
            written.writeheader()
            written.writerows(rows)
        print(table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
        return
    chance = float(np.sum(read.weights[read.events > 0]) / np.sum(read.weights))
    levels = [level for level in arguments.level if level > chance]
    print(f"chance level {chance:.3f}; levels read {levels}")
    if not levels:
        raise SystemExit(f"every level asked for lies at or below the chance level {chance:.3f}")
    if arguments.what == "interaction":
        rows = simulate_interaction(
            read,
            levels=levels,
            teaching=arguments.teaching,
            gains=arguments.gains,
            minimum=arguments.minimum,
            worlds=arguments.worlds,
            resamples=arguments.resamples,
            seed=arguments.seed,
        )
        written_to = arguments.out / INTERACTION_POWER
    else:
        rows = simulate(
            read,
            levels=levels,
            gains=arguments.gains,
            minimum=arguments.minimum,
            worlds=arguments.worlds,
            resamples=arguments.resamples,
            seed=arguments.seed,
        )
        written_to = arguments.out / POWER
    with written_to.open("w", newline="") as handle:
        written = csv.DictWriter(handle, fieldnames=list(rows[0]))
        written.writeheader()
        written.writerows(rows)
    print(table(tuple(rows[0]), [tuple(r.values()) for r in rows]))


if __name__ == "__main__":
    main()
