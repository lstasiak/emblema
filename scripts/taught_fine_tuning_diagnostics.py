"""What fine-tuning on taught events does on a detection task, and how far the task lets anyone go.

Four diagnostics beside the pilot in ``taught_fine_tuning_pilot.py``; none of them judges a claim.
They compare no rule, select nothing and leave the pilot's reading as it stands.

- ``control``: fine-tuning where the answer is easy, to tell a broken recipe from a hard task. The
  long events of the fitting side teach their first halves and are read on their second halves,
  against the normal windows of other months.
- ``curve``: the same cells as the pilot's at fewer steps, each a run of its own, to see whether the
  full run overfits what it was taught.
- ``ceiling``: every event of the fitting side taught, as the most labels the task could give, with
  the frozen detectors and the trees beside.
- ``classes``: each measured cell of the pilot read apart on the events of the classes it was taught
  and on the rest, each half as a gain over the same encoder untaught, adopted from a campaign.

``measure`` touches torch and writes ``cells.csv`` and ``answers/``; ``read`` turns them into tables
and prints them, so a trial sends that output elsewhere.

    uv run --env-file .env.r2 scripts/taught_fine_tuning_diagnostics.py measure
        --task esa-mission1-events --task-id UUID --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --weights base2 KEY CHECKSUM --out DIR --rate 0.001
        [--sections control curve ceiling] [--seeds 1 2 3] [--steps 200 500] [--events 5 10]
        [--ceiling-events 99] [--least-windows 20] [--device mps] [--trial]
    uv run scripts/taught_fine_tuning_diagnostics.py classes --task esa-mission1-events
        --task-id UUID --pilot DIR --untaught DIR --first 2000-01 --last 2006-12 --out DIR
    uv run scripts/taught_fine_tuning_diagnostics.py read --dir DIR [--beside PILOT_DIR]
    uv run scripts/taught_fine_tuning_diagnostics.py checks --dir DIR --beside PILOT_DIR

``checks`` asks after the run whether the readings can be trusted: what a detector of known quality
reaches, what the same scores reach with their link to the events cut, and how each length differs
from the full run.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from uuid import UUID

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.detection.taught_events import TaughtEvents
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.reporting import table
from scripts.taught_class_report import OTHER_HALF, TAUGHT_HALF
from scripts.taught_fine_tuning_pilot import (
    FINE_TUNED,
    INITIALISATION,
    SHARE,
    Bench,
    TunedCell,
    bench_of,
    cell_name,
    committed_tree,
    done_cells,
    read_answers,
    record,
    refuse_another_commit,
    taught_spans,
    tuned_cell,
    write_answers,
)

CONTROL = "control"
CURVE = "curve"
CEILING = "ceiling"
DIRECTION = "direction"
TREES = "trees"
UNTAUGHT = "untaught"
ROTATIONS = 40
KNOWN_LIFTS = (0.0, 3.0, 6.0, 50.0)
SECTIONS = (CONTROL, CURVE, CEILING)
CLASS_COLUMNS = (
    "arm",
    "encoder",
    "events",
    "seed",
    "half",
    "scored_events",
    "area",
    "area_untaught",
    "gain",
    "recall",
)


@dataclass(frozen=True, kw_only=True)
class ControlSides:
    """The long events cut in two, and the normal windows cut by month.

    Attributes:
        teach: The windows a control learns from: the long events' first halves, and normal windows.
        taught: The long events' first halves.
        read: The long events' second halves, and the normal windows of the other months.
        events: How many events are long enough.
    """

    teach: tuple[LabelledWindow, ...]
    taught: tuple[LabelledWindow, ...]
    read: tuple[LabelledWindow, ...]
    events: int


def control_sides(pool: Iterable[LabelledWindow], *, least_windows: int) -> ControlSides:
    """Cut the fitting side for a control, so an encoder that can learn events shows it.

    An event with fewer than ``least_windows`` windows leaves both sides, as do the windows of every
    event left out, so a read normal window is not an unmarked part of something taught. A month
    that holds a read window of an event is read whole, its normal windows with it, since a score
    is read against the other windows of its month and an event alone in one would be read against
    itself. Normal windows teach only in the remaining months of even place.

    Raises:
        SystemExit: If no event is long enough, or no month is left to teach normal windows.
    """
    windows = list(pool)
    spans: dict[int, list[LabelledWindow]] = defaultdict(list)
    for window in windows:
        if AnomalyScheme.is_event(window.target):
            spans[int(window.target)].append(window)
    long_events = {event: span for event, span in spans.items() if len(span) >= least_windows}
    if not long_events:
        raise SystemExit(f"no event holds {least_windows} windows")
    taught: list[LabelledWindow] = []
    held: list[LabelledWindow] = []
    for span in long_events.values():
        ordered = sorted(span, key=lambda w: (str(w.window.unit), w.window.position))
        taught += ordered[: len(ordered) // 2]
        held += ordered[len(ordered) // 2 :]
    read_whole = {str(w.window.unit) for w in held}
    months = sorted({str(w.window.unit) for w in windows})
    teaching_months = set(months[::2]) - read_whole
    if not teaching_months:
        raise SystemExit("every month of even place holds a read window: no normal window teaches")
    normal = [w for w in windows if not AnomalyScheme.is_event(w.target)]
    return ControlSides(
        teach=(*taught, *(w for w in normal if str(w.window.unit) in teaching_months)),
        taught=tuple(taught),
        read=(*held, *(w for w in normal if str(w.window.unit) not in teaching_months)),
        events=len(long_events),
    )


def split_by_class(
    predictions: Iterable[WindowPrediction], classes: Mapping[int, str], taught: frozenset[str]
) -> dict[str, list[WindowPrediction]]:
    """The answers in two halves, each the normal windows plus the events of some classes."""
    split: dict[str, list[WindowPrediction]] = {TAUGHT_HALF: [], OTHER_HALF: []}
    for each in predictions:
        if each.target == AnomalyScheme.NORMAL:
            split[TAUGHT_HALF].append(each)
            split[OTHER_HALF].append(each)
        else:
            half = TAUGHT_HALF if classes[int(each.target)] in taught else OTHER_HALF
            split[half].append(each)
    return split


def class_rows(
    cells: Iterable[Mapping[str, str]],
    directory: Path,
    untaught: Path,
    classes: Mapping[int, str],
    taught_classes: Mapping[tuple[int, int], frozenset[str]],
) -> list[tuple[object, ...]]:
    """One row per cell and half, a half's area read as a gain over the same encoder untaught.

    The area of a half grows with how many events it holds, so two halves are compared by what
    teaching added to each and not by their areas. A half holding no event is left out.

    Raises:
        SystemExit: If an untaught cell scored other windows than the cell it is set against.
    """
    rows: list[tuple[object, ...]] = []
    for cell in cells:
        events, seed = int(cell["events"]), int(cell["seed"])
        answers = read_answers(directory / "answers" / f"{cell['cell']}.csv")
        name = cell_name(UNTAUGHT, cell["backbone"], 0, seed)
        untaught_answers = read_answers(untaught / "answers" / f"{name}.csv")
        if {(str(a.window.unit), a.window.position, a.target) for a in answers} != {
            (str(a.window.unit), a.window.position, a.target) for a in untaught_answers
        }:
            raise SystemExit(f"{name} scored other windows than {cell['cell']}")
        taught = taught_classes[(events, seed)]
        untaught_halves = split_by_class(untaught_answers, classes, taught)
        for half, predictions in split_by_class(answers, classes, taught).items():
            scored = {int(each.target) for each in predictions if each.target}
            if not scored:
                continue
            ranking = AnomalyRanking.of(predictions)
            before = AnomalyRanking.of(untaught_halves[half]).average_precision
            rows.append(
                (
                    cell["arm"],
                    cell["backbone"],
                    events,
                    seed,
                    half,
                    len(scored),
                    ranking.average_precision,
                    before,
                    ranking.average_precision - before,
                    ranking.event_recall_at_false_alarm_share(SHARE),
                )
            )
    return rows


def class_table(rows: Iterable[Sequence[object]]) -> str:
    """Each arm's gain over untaught on the taught classes and on the rest, over the seeds."""
    gains: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        gains[(str(row[0]), int(str(row[2])), str(row[1]))][str(row[4])].append(float(str(row[8])))
    return table(
        ("Arm", "Events", "Encoder", "Gain, taught classes", "Gain, others", "Difference"),
        [
            (
                arm,
                str(events),
                encoder,
                f"{fmean(sides[TAUGHT_HALF]):+.3f}",
                f"{fmean(sides[OTHER_HALF]):+.3f}",
                f"{fmean(sides[TAUGHT_HALF]) - fmean(sides[OTHER_HALF]):+.3f}",
            )
            for (arm, events, encoder), sides in sorted(gains.items())
            if sides[TAUGHT_HALF] and sides[OTHER_HALF]
        ],
    )


def rankings_of(
    directory: Path, cells: Iterable[Mapping[str, str]], *, named_by_steps: bool = False
) -> dict[tuple[str, int], dict[str, dict[int, AnomalyRanking]]]:
    """Every cell's ranking, by arm and count, encoder and seed; the arm named by steps if asked."""
    found: dict[tuple[str, int], dict[str, dict[int, AnomalyRanking]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for cell in cells:
        ranking = AnomalyRanking.of(read_answers(directory / "answers" / f"{cell['cell']}.csv"))
        arm = f"{cell['arm']}-steps{cell['steps']}" if named_by_steps else cell["arm"]
        found[(arm, int(cell["events"]))][cell["backbone"]][int(cell["seed"])] = ranking
    return found


def cells_of(directory: Path) -> list[dict[str, str]]:
    with (directory / "cells.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_tables(directory: Path, *, beside: Path | None = None) -> str:
    """Every arm's area and recall, and pretrained less initialisation where both ran.

    A description: no threshold is applied and nothing passes. ``beside`` names the pilot's
    directory, whose fine-tuning cells stand beside the curve as its full length.

    Raises:
        SystemExit: If no cell was measured.
    """
    cells = cells_of(directory)
    if not cells:
        raise SystemExit(f"no cell in {directory}")
    found = rankings_of(directory, cells)
    if beside is not None:
        full = [cell for cell in cells_of(beside) if cell["arm"].startswith(FINE_TUNED)]
        found.update(rankings_of(beside, full, named_by_steps=True))
    areas = table(
        ("Arm", "Events", "Encoder", "Area per seed", "Area", "Recall @1 %"),
        [
            (
                arm,
                str(events),
                encoder,
                " ".join(f"{side[seed].average_precision:.3f}" for seed in sorted(side)),
                f"{fmean(each.average_precision for each in side.values()):.3f}",
                f"{fmean(e.event_recall_at_false_alarm_share(SHARE) for e in side.values()):.3f}",
            )
            for (arm, events), sides in sorted(found.items())
            for encoder, side in sorted(sides.items())
        ],
    )
    bootstrap = PairedUnitBootstrap()
    gains = []
    for (arm, events), sides in sorted(found.items()):
        if INITIALISATION not in sides or len(sides) < 2:
            continue
        base = sides[INITIALISATION]
        names = sorted(name for name in sides if name != INITIALISATION)
        seeds = sorted(base)
        compared = bootstrap.compare(
            PairedAnomalyRankings(
                control=tuple(base[seed] for seed in seeds) * len(names),
                candidate=tuple(sides[name][seed] for name in names for seed in seeds),
            )
        )
        gains.append(
            (
                arm,
                str(events),
                f"{compared.reduction:+.3f} [{compared.interval.low:+.3f}; "
                f"{compared.interval.high:+.3f}]",
            )
        )
    text = areas + "\n\n" + table(("Arm", "Events", "Pretrained less initialisation"), gains)
    (directory / "tables.md").write_text(text + "\n")
    return text


def _measure_control(
    bench: Bench, out: Path, seeds: Sequence[int], rate: float, least_windows: int
) -> None:  # pragma: no cover - environment
    table_path = out / "cells.csv"
    done = done_cells(table_path)
    sides = control_sides(bench.pool, least_windows=least_windows)
    laid_teach = bench.layout.lay(bench.block.at([w.window.position for w in sides.teach]))
    laid_read = bench.layout.lay(bench.block.at([w.window.position for w in sides.read]))
    spans, others = taught_spans(sides.teach, sides.taught)
    arm = f"{CONTROL}-{FINE_TUNED}{rate}"
    rated = replace(bench.schedule, learning_rate=rate)
    print(
        json.dumps(
            {"control_events": sides.events, "taught": len(sides.taught), "read": len(sides.read)}
        ),
        flush=True,
    )
    for seed in seeds:
        for name, backbone in bench.arms:
            cell = cell_name(arm, name, sides.events, seed)
            if cell in done:
                continue
            tuned = tuned_cell(
                bench,
                name,
                backbone,
                schedule=rated,
                seed=seed,
                spans=spans,
                others=others,
                laid_teach=laid_teach,
                laid_read=laid_read,
                read=sides.read,
            )
            _keep(
                out, cell, arm, name, sides.events, seed, len(sides.taught), rated.min_steps, tuned
            )
    measure_frozen(
        bench,
        out,
        seeds,
        arm=f"{CONTROL}-{DIRECTION}",
        events=sides.events,
        fitting=[w.window for w in sides.teach],
        read=sides.read,
        taught=sides.taught,
        trees=False,
    )


def _keep(
    out: Path,
    cell: str,
    arm: str,
    name: str,
    events: int,
    seed: int,
    taught: int,
    steps: int,
    tuned: TunedCell,
) -> None:  # pragma: no cover - environment
    write_answers(out / "answers" / f"{cell}.csv", tuned.predictions)
    record(
        out / "cells.csv",
        (
            cell,
            arm,
            name,
            events,
            seed,
            taught,
            steps,
            tuned.seconds,
            tuned.early,
            tuned.late,
            tuned.moved,
        ),
    )
    print(f"{cell}: {tuned.seconds:.0f} s", flush=True)


def measure_frozen(
    bench: Bench,
    out: Path,
    seeds: Sequence[int],
    *,
    arm: str,
    events: int,
    fitting: Sequence[TaskWindow],
    read: Sequence[LabelledWindow],
    taught: Sequence[LabelledWindow] | None,
    trees: bool,
) -> None:  # pragma: no cover - environment
    """The frozen direction under every encoder, and the trees, by the runtimes campaigns use."""
    from emblema.entrypoints.workers.known_detectors import KnownDetectors
    from emblema.evaluation.adapters.detection.summaries_detection_runtime import (
        SummariesDetectionRuntime,
    )
    from emblema.evaluation.adapters.torch.torch_detection_runtime import TorchDetectionRuntime
    from emblema.evaluation.contracts.identifiers import CandidateRef

    table_path = out / "cells.csv"
    done = done_cells(table_path)
    architecture = next(iter(bench.weights.values()))
    for seed in seeds:
        drawn = (
            taught
            if taught is not None
            else TaughtEvents.drawn(bench.registered, bench.pool, events=events, seed=seed).windows
        )
        for name, backbone in bench.arms:
            cell = cell_name(arm, name, events, seed)
            if cell in done:
                continue
            started = time.perf_counter()
            which = "untrained_knn" if name == INITIALISATION else "frozen_knn"
            recipe = (
                KnownDetectors.catalogue(bench.weights.get(name, architecture))
                .arm_of(CandidateRef(f"{which}@events={events},reference=unit"))
                .recipe(seed=seed)
            )
            outcome = TorchDetectionRuntime(backbone, bench.blocks, device=bench.device).fit(
                recipe, bench.task, fitting, read, taught=drawn, retain=False
            )
            write_answers(out / "answers" / f"{cell}.csv", outcome.predictions)
            seconds = time.perf_counter() - started
            record(table_path, (cell, arm, name, events, seed, len(drawn), 0, seconds, "", "", ""))
            print(f"{cell}: {seconds:.0f} s", flush=True)
        cell = cell_name(TREES, "summaries", events, seed)
        if trees and cell not in done:
            started = time.perf_counter()
            recipe = (
                KnownDetectors.catalogue(architecture)
                .arm_of(CandidateRef(f"summaries_knn@events={events},head=trees"))
                .recipe(seed=seed)
            )
            outcome = SummariesDetectionRuntime(bench.blocks).fit(
                recipe, bench.task, fitting, read, taught=drawn, retain=False
            )
            write_answers(out / "answers" / f"{cell}.csv", outcome.predictions)
            seconds = time.perf_counter() - started
            record(
                table_path,
                (cell, TREES, "summaries", events, seed, len(drawn), 0, seconds, "", "", ""),
            )
            print(f"{cell}: {seconds:.0f} s", flush=True)


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """The sections asked for, on the device, cells already answered skipped."""
    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = "trial" if arguments.trial else committed_tree()
    refuse_another_commit(out / "conditions.json", commit)
    bench = bench_of(arguments)
    every_event = len({int(w.target) for w in bench.pool if w.target})
    if CEILING in arguments.sections and arguments.ceiling_events != every_event:
        raise SystemExit(
            f"the ceiling teaches all {every_event} events: {arguments.ceiling_events}"
        )
    conditions = {
        "commit": commit,
        "encoders": list(bench.weights),
        "seeds": arguments.seeds,
        "rate": arguments.rate,
        "steps": bench.schedule.min_steps,
        "curve_steps": arguments.steps,
        "batch": bench.schedule.batch_size,
        "events": arguments.events,
        "ceiling_events": arguments.ceiling_events,
        "least_windows": arguments.least_windows,
        "device": arguments.device,
    }
    refuse_other_conditions(out / "conditions.json", conditions)
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
    rated = replace(bench.schedule, learning_rate=arguments.rate)
    if CONTROL in arguments.sections:
        _measure_control(bench, out, arguments.seeds, arguments.rate, arguments.least_windows)
    plans = []
    if CURVE in arguments.sections:
        plans += [
            (f"{FINE_TUNED}{arguments.rate}-steps{steps}", steps, events)
            for steps in arguments.steps
            for events in arguments.events
        ]
    if CEILING in arguments.sections:
        plans.append(
            (f"{CEILING}-{FINE_TUNED}{arguments.rate}", rated.min_steps, arguments.ceiling_events)
        )
    if plans:
        done = done_cells(out / "cells.csv")
        laid_pool = bench.layout.lay(bench.block.at([w.window.position for w in bench.pool]))
        laid_read = bench.layout.lay(bench.block.at([w.window.position for w in bench.read]))
        for arm, steps, events in plans:
            for seed in arguments.seeds:
                taught = TaughtEvents.drawn(
                    bench.registered, bench.pool, events=events, seed=seed
                ).windows
                spans, others = taught_spans(bench.pool, taught)
                for name, backbone in bench.arms:
                    cell = cell_name(arm, name, events, seed)
                    if cell in done:
                        continue
                    tuned = tuned_cell(
                        bench,
                        name,
                        backbone,
                        schedule=replace(rated, min_steps=steps),
                        seed=seed,
                        spans=spans,
                        others=others,
                        laid_teach=laid_pool,
                        laid_read=laid_read,
                        read=bench.read,
                    )
                    _keep(out, cell, arm, name, events, seed, len(taught), steps, tuned)
        # The frozen runtime lays both sides again, so these need not stay in memory beside them.
        del laid_pool, laid_read
    if CEILING in arguments.sections:
        measure_frozen(
            bench,
            out,
            arguments.seeds,
            arm=f"{CEILING}-{DIRECTION}",
            events=arguments.ceiling_events,
            fitting=[w.window for w in bench.pool],
            read=bench.read,
            taught=None,
            trees=True,
        )


def refuse_other_conditions(path: Path, conditions: Mapping[str, object]) -> None:
    """Refuse to add cells to a run begun under other conditions, as they share its cell names.

    Raises:
        SystemExit: If ``path`` records any other condition.
    """
    if path.exists():
        begun = json.loads(path.read_text())
        differing = sorted(key for key in conditions if begun.get(key) != conditions[key])
        if differing:
            raise SystemExit(f"this run began under other conditions: {differing}")


def classes(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Read the pilot's cells by the classes each was taught; a table, nothing measured."""
    from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS, EsaAdRawCorpusReader
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
    from emblema.evaluation.contracts.identifiers import TaskId
    from scripts.raw_corpora import raw_root
    from scripts.taught_class_report import FittingEvents

    task = KnownTasks.named(arguments.task)
    mission = task.unit_prefix.rstrip("/")
    root = raw_root("esa_ad")
    if root is None:
        raise SystemExit("the ESA Anomaly Dataset is not downloaded")
    truth = EsaAdRawGroundTruth(
        root,
        channels=EsaAdRawCorpusReader.POLICY.channels,
        origin_seconds=ORIGIN_SECONDS,
        window_hours=KnownGroundTruths.ESA_AD_RAW_WINDOW_HOURS,
    )
    fitting = FittingEvents.of(truth, mission, arguments.first, arguments.last)
    registered = TaskId(UUID(arguments.task_id))
    cells = [row for row in cells_of(arguments.pilot) if row["arm"] != TREES]
    taught = {
        (int(cell["events"]), int(cell["seed"])): fitting.taught_classes(
            registered, int(cell["events"]), int(cell["seed"])
        )
        for cell in cells
    }
    rows = class_rows(cells, arguments.pilot, arguments.untaught, fitting.classes, taught)
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "classes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CLASS_COLUMNS)
        writer.writerows(rows)
    (arguments.out / "classes.md").write_text(class_table(rows) + "\n")
    print(f"{len(rows)} rows in {arguments.out / 'classes.csv'}")


def with_known_lift(
    answers: Sequence[WindowPrediction], lift: float, *, seed: int
) -> AnomalyRanking:
    """A detector of known quality over these windows: noise, with events ``lift`` deviations up.

    Scored as a cell is, each month against itself, so a large lift shows the most this reading
    lets any detector reach, and no lift what scores with no link to anything reach.
    """
    from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm

    raw = np.random.default_rng(seed).normal(size=len(answers)) + np.array(
        [lift if each.target else 0.0 for each in answers]
    )
    units = [str(each.window.unit) for each in answers]
    alarm = ScoreAlarm.set_over(raw, contamination=SHARE, units=units)
    return AnomalyRanking.of(
        [
            WindowPrediction(window=each.window, target=each.target, predicted=score)
            for each, score in zip(answers, alarm.above(raw, units=units).tolist(), strict=True)
        ]
    )


def rotated_within_months(
    answers: Sequence[WindowPrediction], draws: random.Random
) -> list[WindowPrediction]:
    """The same scores turned by a random offset within each month, cutting their link to events.

    Neighbouring scores stay neighbours, so a run of high scores is kept: scores drawn afresh for
    every window would reach a long event far more often than a real detector's could by chance.
    """
    by_month: dict[str, list[WindowPrediction]] = defaultdict(list)
    for each in answers:
        by_month[str(each.window.unit)].append(each)
    rotated = []
    for month in by_month.values():
        ordered = sorted(month, key=lambda each: each.window.position)
        offset = draws.randrange(1, len(ordered)) if len(ordered) > 1 else 0
        rotated += [
            replace(each, predicted=ordered[(at + offset) % len(ordered)].predicted)
            for at, each in enumerate(ordered)
        ]
    return rotated


def _baselines_table(directory: Path, beside: Path) -> str:
    draws = random.Random(7)
    chosen = [
        (place, cell)
        for place in (directory, beside)
        for cell in cells_of(place)
        if cell["seed"] == "1" and cell["arm"] != TREES and cell["arm"] != DIRECTION
    ]
    rows = []
    for place, cell in chosen:
        answers = read_answers(place / "answers" / f"{cell['cell']}.csv")
        real = AnomalyRanking.of(answers)
        turned = [
            AnomalyRanking.of(rotated_within_months(answers, draws)) for _ in range(ROTATIONS)
        ]
        rows.append(
            (
                cell["cell"],
                f"{real.average_precision:.3f}",
                f"{real.event_recall_at_false_alarm_share(SHARE):.3f}",
                f"{fmean(each.average_precision for each in turned):.4f}",
                f"{fmean(each.event_recall_at_false_alarm_share(SHARE) for each in turned):.3f}",
            )
        )
    return table(("Cell", "Area", "Recall", "Rotated area", "Rotated recall"), rows)


def _known_lifts_table(directory: Path, cell_names: Mapping[str, str]) -> str:
    rows = []
    for side, name in cell_names.items():
        answers = read_answers(directory / "answers" / f"{name}.csv")
        for lift in KNOWN_LIFTS:
            ranking = with_known_lift(answers, lift, seed=0)
            rows.append(
                (
                    side,
                    f"{lift:.0f}",
                    f"{ranking.average_precision:.3f}",
                    f"{ranking.event_recall_at_false_alarm_share(SHARE):.3f}",
                )
            )
    return table(("Windows read", "Lift", "Area", "Recall"), rows)


def _lengths_table(directory: Path, beside: Path) -> str:
    bootstrap = PairedUnitBootstrap()
    full = rankings_of(beside, [c for c in cells_of(beside) if c["arm"].startswith(FINE_TUNED)])
    rows = []
    shorter = [c for c in cells_of(directory) if c["arm"].startswith(FINE_TUNED)]
    for (arm, events), sides in sorted(rankings_of(directory, shorter).items()):
        whole = full[(arm.split("-steps")[0], events)]
        for name in (*sorted(sides), "every encoder"):
            names = sorted(sides) if name == "every encoder" else [name]
            compared = bootstrap.compare(
                PairedAnomalyRankings(
                    control=tuple(
                        whole[each][seed] for each in names for seed in sorted(whole[each])
                    ),
                    candidate=tuple(
                        sides[each][seed] for each in names for seed in sorted(whole[each])
                    ),
                )
            )
            rows.append(
                (
                    arm,
                    str(events),
                    name,
                    f"{compared.reduction:+.3f} [{compared.interval.low:+.3f}; "
                    f"{compared.interval.high:+.3f}]",
                )
            )
    return table(("Arm", "Events", "Encoder", "Less the full run"), rows)


def _training_table(directory: Path, beside: Path) -> str:
    gathered: dict[tuple[str, int, int, str], list[Mapping[str, str]]] = defaultdict(list)
    for cell in cells_of(directory) + cells_of(beside):
        if cell["last_loss"]:
            key = (cell["arm"], int(cell["events"]), int(cell["steps"]), cell["backbone"])
            gathered[key].append(cell)
    return table(
        ("Arm", "Events", "Steps", "Encoder", "Late loss", "Moved"),
        [
            (
                arm,
                str(events),
                str(steps),
                encoder,
                f"{fmean(float(each['last_loss']) for each in cells):.3f}",
                f"{fmean(float(each['moved']) for each in cells):.3f}",
            )
            for (arm, events, steps, encoder), cells in sorted(gathered.items())
        ],
    )


def checks_tables(directory: Path, beside: Path) -> str:
    """Whether the readings can be trusted, asked after the run; nothing here was declared."""
    text = "\n\n".join(
        (
            _known_lifts_table(
                directory,
                {
                    "control": f"{CONTROL}-{DIRECTION}-base1-k36-s1",
                    "unread months": f"{CEILING}-{DIRECTION}-base1-k99-s1",
                },
            ),
            _baselines_table(directory, beside),
            _lengths_table(directory, beside),
            _training_table(directory, beside),
        )
    )
    (directory / "checks.md").write_text(text + "\n")
    return text


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True)
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument("--weights", nargs=3, action="append", required=True)
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--rate", type=float, required=True)
    measuring.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    measuring.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    measuring.add_argument("--steps", type=int, nargs="+", default=[200, 500])
    measuring.add_argument("--events", type=int, nargs="+", default=[5, 10])
    measuring.add_argument("--ceiling-events", type=int, default=99)
    measuring.add_argument("--least-windows", type=int, default=20)
    measuring.add_argument("--device", default="mps")
    measuring.add_argument(
        "--trial", action="store_true", help="run before the declaration is committed"
    )
    by_class = commands.add_parser("classes")
    by_class.add_argument("--task", required=True)
    by_class.add_argument("--task-id", required=True)
    by_class.add_argument("--pilot", type=Path, required=True)
    by_class.add_argument("--untaught", type=Path, required=True)
    by_class.add_argument("--first", required=True)
    by_class.add_argument("--last", required=True)
    by_class.add_argument("--out", type=Path, required=True)
    reading = commands.add_parser("read")
    reading.add_argument("--dir", type=Path, required=True)
    reading.add_argument(
        "--beside", type=Path, help="the pilot's directory, for the curve's full length"
    )
    checking = commands.add_parser("checks")
    checking.add_argument("--dir", type=Path, required=True)
    checking.add_argument("--beside", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure, read by class, or turn what was measured into tables."""
    arguments = parse_arguments(argv)
    if arguments.command == "measure":  # pragma: no cover - environment
        measure(arguments)
    elif arguments.command == "classes":  # pragma: no cover - environment
        classes(arguments)
    elif arguments.command == "checks":  # pragma: no cover - environment
        print(checks_tables(arguments.dir, arguments.beside))
    else:
        print(read_tables(arguments.dir, beside=arguments.beside))


if __name__ == "__main__":  # pragma: no cover
    main()
