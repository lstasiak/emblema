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

``structure`` writes one row per validation window of the task, with its unit and the event it
overlaps, from the raw files under ``data/raw``; ``simulate`` writes ``power.csv`` and prints
the table a note pastes. The areas are computed by the same sweep the harness reads, checked
against it on a small case.
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

STRUCTURE, POWER = "structure.csv", "power.csv"
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
        held = _months(mission, first, last)
        windows = [
            TaskWindow(unit=UnitKey(key), position=position, ends_at=hour)
            for key in _validation(task, held)
            for position, hour in enumerate(_hours_of(key))
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


def _months(mission: str, first: str, last: str) -> list[str]:
    year, month = (int(part) for part in first.split("-"))
    last_year, last_month = (int(part) for part in last.split("-"))
    keys = []
    while (year, month) <= (last_year, last_month):
        keys.append(f"{mission}/{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return keys


def _hours_of(key: str) -> list[float]:
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
    read = Structure.read(arguments.structure)
    chance = float(np.sum(read.weights[read.events > 0]) / np.sum(read.weights))
    levels = [level for level in arguments.level if level > chance]
    print(f"chance level {chance:.3f}; levels read {levels}")
    if not levels:
        raise SystemExit(f"every level asked for lies at or below the chance level {chance:.3f}")
    rows = simulate(
        read,
        levels=levels,
        gains=arguments.gains,
        minimum=arguments.minimum,
        worlds=arguments.worlds,
        resamples=arguments.resamples,
        seed=arguments.seed,
    )
    with (arguments.out / POWER).open("w", newline="") as handle:
        written = csv.DictWriter(handle, fieldnames=list(rows[0]))
        written.writeheader()
        written.writerows(rows)
    print(table(("level", "gain", "minimum", "worlds", "power"), [tuple(r.values()) for r in rows]))


if __name__ == "__main__":
    main()
