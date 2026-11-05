"""How much this network reaches on the first mission when it is given every label, not a pretext.

The frozen direction taught every event of the fitting side reaches about 0.26 with or without
pretraining, and fine-tuning read at one length reaches 0.09. Neither says how much room the task
leaves a network of this shape. This reference trains the same network from its initialisation on
every event, with the length and rate chosen where they may be chosen: on the later months of the
fitting side, never on the months read. The pretrained encoders run beside it under the same rule.

Three sections, each a run of its own:

- ``select``: the earlier fitting months teach, each rate and length a run with its own schedule;
  every run scores the selection months, which choose, and the validation months, which are kept
  unread until the refit is whole.
- ``bars``: the frozen direction taught the same events, scored on the selection months, so the
  choice can say whether any length reaches it there.
- ``refit``: at the rate and length ``choose`` picked for each encoder, the whole fitting side
  teaches and the validation months are read: the reference itself.

``measure`` touches torch and writes ``cells*.csv`` and ``answers/``; a lane ``i/n`` takes every
n-th cell so several processes share one grid. ``choose`` reads the selection months alone and
writes ``chosen.json``; ``read`` refuses to open the validation months before every refit cell is
in, then sets the reference against the bars adopted from the diagnostics of the few-event pilot.

    uv run --env-file .env.r2 scripts/telemetry_supervised_reference.py measure
        --task esa-mission1-events --task-id UUID --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --weights base2 KEY CHECKSUM --out DIR
        --rates 0.001 0.0003 --lengths 500 1000 2000 4000 8000 16000 --select-from 2004-01
        [--section select|bars|refit] [--seeds 1 2 3] [--lane 1/3] [--device mps] [--trial]
    uv run scripts/telemetry_supervised_reference.py choose --dir DIR [--partial]
    uv run scripts/telemetry_supervised_reference.py adopt --dir DIR --from DIAGNOSTICS_DIR
        --arm ceiling-direction direction --arm trees trees
    uv run scripts/telemetry_supervised_reference.py classes --task esa-mission1-events
        --dir DIR --first 2000-01 --last 2006-12 [--partial]
    uv run scripts/telemetry_supervised_reference.py read --dir DIR [--partial]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.reporting import table
from scripts.taught_fine_tuning_diagnostics import measure_frozen
from scripts.taught_fine_tuning_pilot import (
    CELL_COLUMNS,
    FINE_TUNED,
    INITIALISATION,
    LEAST_GAIN,
    SHARE,
    Bench,
    bench_of,
    cell_name,
    committed_tree,
    fine_tune,
    read_answers,
    record,
    scored,
    taught_spans,
    write_answers,
)

SELECT = "select"
BARS = "bars"
REFIT = "refit"
DIRECTION = "direction"
SECTIONS = (SELECT, BARS, REFIT)
# Where the frozen direction's cells on the selection months live, apart from the fine-tuning's.
BARS_DIRECTORY = "bars"
SELECTION_SUFFIX = "@selection"
# Lengths whose means on the selection months lie this close to the best are one choice.
WITHIN = 0.01
SEEN_HALF, UNSEEN_HALF = "seen classes", "unseen classes"
CLASS_COLUMNS = ("arm", "encoder", "seed", "half", "scored_events", "area", "recall")


def month_of(window: LabelledWindow) -> str:
    """The month a window's unit names, after the mission."""
    return str(window.window.unit).rpartition("/")[2]


@dataclass(frozen=True, kw_only=True)
class TimeSides:
    """The fitting side cut by month: the earlier months teach, the later ones choose.

    Attributes:
        teach: The earlier months' windows, every event among them taught.
        taught: The event windows among those.
        read: The later months' windows, less the windows of events that began earlier.
        events_taught: How many events the earlier months hold.
        events_read: How many events are read on the later months.
        events_dropped: How many events lie on both sides, their later windows left unscored.
    """

    teach: tuple[LabelledWindow, ...]
    taught: tuple[LabelledWindow, ...]
    read: tuple[LabelledWindow, ...]
    events_taught: int
    events_read: int
    events_dropped: int


def time_sides(pool: Iterable[LabelledWindow], *, select_from: str) -> TimeSides:
    """Cut the fitting side at a month, so a length is chosen on months that taught nothing.

    An event with windows on both sides would be read partly against itself, as the control of
    the diagnostics was before its cut: its later windows are left unscored and its earlier ones
    teach.

    Raises:
        SystemExit: If either side holds no event.
    """
    windows = list(pool)
    teach = [w for w in windows if month_of(w) < select_from]
    later = [w for w in windows if month_of(w) >= select_from]
    taught_events = {int(w.target) for w in teach if AnomalyScheme.is_event(w.target)}
    later_events = {int(w.target) for w in later if AnomalyScheme.is_event(w.target)}
    straddling = taught_events & later_events
    read = [w for w in later if int(w.target) not in straddling]
    if not taught_events or not (later_events - straddling):
        raise SystemExit(f"a cut at {select_from} leaves a side without an event")
    return TimeSides(
        teach=tuple(teach),
        taught=tuple(w for w in teach if AnomalyScheme.is_event(w.target)),
        read=tuple(read),
        events_taught=len(taught_events),
        events_read=len(later_events - straddling),
        events_dropped=len(straddling),
    )


def arm_name(section: str, rate: float, steps: int) -> str:
    return f"{section}-{FINE_TUNED}{rate}-steps{steps}"


def split_arm(arm: str) -> tuple[str, float, int]:
    """The section, rate and steps an arm's name states.

    Raises:
        ValueError: If the name is not one ``arm_name`` wrote.
    """
    section, separator, rest = arm.partition(f"-{FINE_TUNED}")
    if not separator or "-steps" not in rest:
        raise ValueError(f"{arm} names no fine-tuning at a length")
    rate, steps = rest.split("-steps")
    return section, float(rate), int(steps)


@dataclass(frozen=True, kw_only=True)
class Plan:
    """One fine-tuning cell the grid holds.

    Attributes:
        section: Which section it belongs to.
        rate: The peak rate of its schedule.
        steps: How many optimiser steps it takes.
        encoder: Which encoder it starts from.
        seed: The seed that draws its start and its batches.
        events: How many events it is taught, for its name.
    """

    section: str
    rate: float
    steps: int
    encoder: str
    seed: int
    events: int

    @property
    def cell(self) -> str:
        return cell_name(
            arm_name(self.section, self.rate, self.steps), self.encoder, self.events, self.seed
        )


def select_plans(
    rates: Sequence[float],
    lengths: Sequence[int],
    encoders: Sequence[str],
    seeds: Sequence[int],
    *,
    events: int,
) -> list[Plan]:
    """Every selection cell, the longest first so lanes share the cost alike."""
    return [
        Plan(section=SELECT, rate=rate, steps=steps, encoder=encoder, seed=seed, events=events)
        for steps in sorted(lengths, reverse=True)
        for rate in rates
        for encoder in encoders
        for seed in seeds
    ]


def lane_of(plans: Sequence[Plan], lane: str | None) -> list[Plan]:
    """The cells a lane ``i/n`` takes: every n-th from the i-th; all of them without a lane.

    Raises:
        SystemExit: If the lane is not ``i/n`` with 1 ≤ i ≤ n.
    """
    if lane is None:
        return list(plans)
    try:
        at, count = (int(part) for part in lane.split("/"))
    except ValueError:
        raise SystemExit(f"a lane is i/n, got {lane}") from None
    if not 1 <= at <= count:
        raise SystemExit(f"a lane is i/n with 1 <= i <= n, got {lane}")
    return list(plans[at - 1 :: count])


@dataclass(frozen=True, kw_only=True)
class Choice:
    """The rate and length chosen for one encoder on the selection months.

    Attributes:
        encoder: Which encoder.
        rate: The rate chosen.
        steps: The length chosen.
        mean: Its mean area over the seeds on the selection months.
        per_seed: The area under each seed.
        extend: Whether the chosen length is the grid's longest, so the grid should grow once.
        bar: The frozen direction's mean area on the same months, where it was measured.
    """

    encoder: str
    rate: float
    steps: int
    mean: float
    per_seed: Mapping[int, float]
    extend: bool
    bar: float | None

    @property
    def untrainable(self) -> bool | None:
        """Whether no length reaches the frozen direction on the selection months."""
        return None if self.bar is None else self.mean < self.bar


def choose(
    areas: Mapping[tuple[str, float, int], Mapping[int, float]],
    *,
    lengths: Sequence[int],
    bars: Mapping[str, float],
) -> list[Choice]:
    """Each encoder's rate and length by one rule, from the selection months alone.

    The best mean over seeds names a set: every run within ``WITHIN`` of it. Of those the
    shortest wins, since a cell repeated on the accelerator moves by about that much and a
    shorter run costs less and overfits less; among equals the higher mean, then the higher rate,
    which is the one in force.

    Raises:
        SystemExit: If an encoder's runs hold different seeds.
    """
    by_encoder: dict[str, dict[tuple[float, int], Mapping[int, float]]] = defaultdict(dict)
    for (encoder, rate, steps), seeds in areas.items():
        by_encoder[encoder][(rate, steps)] = seeds
    choices = []
    for encoder, runs in sorted(by_encoder.items()):
        held = {tuple(sorted(seeds)) for seeds in runs.values()}
        if len(held) != 1:
            raise SystemExit(f"{encoder} holds different seeds across its runs: {sorted(held)}")
        means = {run: fmean(seeds.values()) for run, seeds in runs.items()}
        best = max(means.values())
        close = [run for run, mean in means.items() if mean >= best - WITHIN]
        rate, steps = min(close, key=lambda run: (run[1], -means[run], -run[0]))
        choices.append(
            Choice(
                encoder=encoder,
                rate=rate,
                steps=steps,
                mean=means[(rate, steps)],
                per_seed=dict(sorted(runs[(rate, steps)].items())),
                extend=steps == max(lengths),
                bar=bars.get(encoder),
            )
        )
    return choices


def choice_table(choices: Sequence[Choice]) -> str:
    return table(
        ("Encoder", "Rate", "Steps", "Area on selection months", "Per seed", "Direction", "Says"),
        [
            (
                choice.encoder,
                f"{choice.rate:g}",
                str(choice.steps),
                f"{choice.mean:.3f}",
                " ".join(f"{area:.3f}" for area in choice.per_seed.values()),
                "" if choice.bar is None else f"{choice.bar:.3f}",
                ", ".join(
                    word
                    for word, holds in (
                        ("extend the grid", choice.extend),
                        ("untrainable", bool(choice.untrainable)),
                    )
                    if holds
                )
                or "chosen",
            )
            for choice in choices
        ],
    )


def cells_of(directory: Path) -> list[dict[str, str]]:
    """Every cell the directory's lanes recorded."""
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("cells*.csv")):
        with path.open(newline="") as handle:
            rows += list(csv.DictReader(handle))
    return rows


def selection_areas(directory: Path) -> dict[tuple[str, float, int], dict[int, float]]:
    """Every selection cell's area on the selection months, by encoder, rate and length."""
    areas: dict[tuple[str, float, int], dict[int, float]] = defaultdict(dict)
    for cell in cells_of(directory):
        if not cell["arm"].startswith(SELECT):
            continue
        _, rate, steps = split_arm(cell["arm"])
        answers = read_answers(directory / "answers" / f"{cell['cell']}{SELECTION_SUFFIX}.csv")
        areas[(cell["backbone"], rate, steps)][int(cell["seed"])] = AnomalyRanking.of(
            answers
        ).average_precision
    return areas


def selection_bars(directory: Path) -> dict[str, float]:
    """The frozen direction's mean area on the selection months, by encoder, where measured."""
    bars = directory / BARS_DIRECTORY
    if not bars.exists():
        return {}
    areas: dict[str, list[float]] = defaultdict(list)
    for cell in cells_of(bars):
        answers = read_answers(bars / "answers" / f"{cell['cell']}.csv")
        areas[cell["backbone"]].append(AnomalyRanking.of(answers).average_precision)
    return {encoder: fmean(found) for encoder, found in areas.items()}


def missing_selects(directory: Path, conditions: Mapping[str, Any]) -> list[str]:
    """The selection cells the conditions' grid holds and the lanes have not recorded."""
    held = done_cells(directory)
    plans = select_plans(
        conditions["rates"],
        conditions["lengths"],
        conditions["encoders"],
        conditions["seeds"],
        events=conditions["events_taught_in_selection"],
    )
    return sorted(plan.cell for plan in plans if plan.cell not in held)


def missing_bars(directory: Path, conditions: Mapping[str, Any]) -> list[str]:
    """The frozen direction's cells on the selection months the conditions ask for and lack."""
    bars = directory / BARS_DIRECTORY
    held = done_cells(bars) if bars.exists() else set()
    return sorted(
        cell
        for encoder in conditions["encoders"]
        for seed in conditions["seeds"]
        if (
            cell := cell_name(
                f"{SELECT}-{DIRECTION}", encoder, conditions["events_taught_in_selection"], seed
            )
        )
        not in held
    )


def choose_from(directory: Path, *, partial: bool = False) -> list[Choice]:
    """The choices, from the selection months alone, written to ``chosen.json``.

    A lane that gave up would leave one seed out of every run, and a bar that failed would leave
    an encoder without its untrainable reading, so a grid not yet whole is refused unless asked.

    Raises:
        SystemExit: If no selection cell was measured, or a selection cell or a bar is missing
            and ``partial`` is not asked.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    areas = selection_areas(directory)
    if not areas:
        raise SystemExit(f"no selection cell in {directory}")
    if not partial:
        for what, missing in (
            ("selection cells", missing_selects(directory, conditions)),
            ("bars", missing_bars(directory, conditions)),
        ):
            if missing:
                raise SystemExit(f"{len(missing)} {what} are missing, the first {missing[0]}")
    choices = choose(areas, lengths=conditions["lengths"], bars=selection_bars(directory))
    (directory / "chosen.json").write_text(
        json.dumps(
            [
                {
                    "encoder": choice.encoder,
                    "rate": choice.rate,
                    "steps": choice.steps,
                    "mean": choice.mean,
                    "extend": choice.extend,
                    "untrainable": choice.untrainable,
                }
                for choice in choices
            ],
            indent=2,
        )
        + "\n"
    )
    (directory / "chosen.md").write_text(choice_table(choices) + "\n")
    return choices


def refit_plans(directory: Path, seeds: Sequence[int], *, events: int) -> list[Plan]:
    """The refit cells ``chosen.json`` asks for.

    Raises:
        SystemExit: If nothing was chosen, or a choice asks for the grid to grow first.
    """
    path = directory / "chosen.json"
    if not path.exists():
        raise SystemExit("choose first: no chosen.json")
    chosen = json.loads(path.read_text())
    extend = [choice["encoder"] for choice in chosen if choice["extend"]]
    if extend:
        raise SystemExit(f"the grid should grow once before a refit: {extend}")
    return [
        Plan(
            section=REFIT,
            rate=choice["rate"],
            steps=choice["steps"],
            encoder=choice["encoder"],
            seed=seed,
            events=events,
        )
        for choice in chosen
        for seed in seeds
    ]


def refuse_other_conditions(path: Path, conditions: Mapping[str, object]) -> None:
    """Refuse to add cells to a run begun under other conditions; the lengths may only grow.

    Raises:
        SystemExit: If ``path`` records any other condition, or fewer lengths than it had.
    """
    if not path.exists():
        return
    begun = json.loads(path.read_text())
    differing = sorted(
        key for key in conditions if key != "lengths" and begun.get(key) != conditions[key]
    )
    if differing:
        raise SystemExit(f"this run began under other conditions: {differing}")
    lengths = conditions["lengths"]
    if not isinstance(lengths, list) or not set(begun.get("lengths", [])) <= set(lengths):
        raise SystemExit("a run's lengths may grow, not shrink")


def done_cells(directory: Path) -> set[str]:
    return {cell["cell"] for cell in cells_of(directory)}


def lane_table(directory: Path, lane: str | None) -> Path:
    """The table a lane records into, begun if there is none."""
    path = directory / ("cells.csv" if lane is None else f"cells-{lane.replace('/', '-of-')}.csv")
    if not path.exists():
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(CELL_COLUMNS)
    return path


def _measure_fine_tuning(  # pragma: no cover - environment
    bench: Bench,
    directory: Path,
    plans: Sequence[Plan],
    *,
    lane: str | None,
    teach: Sequence[LabelledWindow],
    taught: Sequence[LabelledWindow],
    reads: Mapping[str, Sequence[LabelledWindow]],
) -> None:
    """Each plan's cell, trained once and scored on every side in ``reads``, by its suffix."""
    done = done_cells(directory)
    table_path = lane_table(directory, lane)
    laid_teach = bench.layout.lay(bench.block.at([w.window.position for w in teach]))
    laid_reads = {
        suffix: bench.layout.lay(bench.block.at([w.window.position for w in read]))
        for suffix, read in reads.items()
    }
    spans, others = taught_spans(teach, taught)
    for plan in plans:
        if plan.cell in done:
            continue
        started = time.perf_counter()
        schedule = replace(bench.schedule, learning_rate=plan.rate, min_steps=plan.steps)
        tuned = fine_tune(
            bench,
            plan.encoder,
            bench.backbones.get(plan.encoder, next(iter(bench.backbones.values()))),
            schedule=schedule,
            seed=plan.seed,
            spans=spans,
            others=others,
            laid_teach=laid_teach,
        )
        for suffix, read in reads.items():
            write_answers(
                directory / "answers" / f"{plan.cell}{suffix}.csv",
                scored(bench, tuned.candidate, laid_reads[suffix], read),
            )
        seconds = time.perf_counter() - started
        record(
            table_path,
            (
                plan.cell,
                arm_name(plan.section, plan.rate, plan.steps),
                plan.encoder,
                plan.events,
                plan.seed,
                len(taught),
                plan.steps,
                seconds,
                tuned.early,
                tuned.late,
                tuned.moved,
            ),
        )
        print(f"{plan.cell}: {seconds:.0f} s", flush=True)


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """The section asked for, on the device, cells already answered skipped."""
    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = "trial" if arguments.trial else committed_tree()
    bench = bench_of(arguments)
    sides = time_sides(bench.pool, select_from=arguments.select_from)
    every_event = len({int(w.target) for w in bench.pool if w.target})
    conditions = {
        "commit": commit,
        "encoders": [*bench.weights, INITIALISATION],
        "seeds": arguments.seeds,
        "rates": arguments.rates,
        "lengths": sorted(arguments.lengths),
        "select_from": arguments.select_from,
        "batch": bench.schedule.batch_size,
        "events_taught_in_selection": sides.events_taught,
        "events_on_selection_months": sides.events_read,
        "events_dropped": sides.events_dropped,
        "events_in_pool": every_event,
        "windows_teaching_in_selection": len(sides.teach),
        "windows_on_selection_months": len(sides.read),
        "windows_read": len(bench.read),
        "months_read": len({str(w.window.unit) for w in bench.read}),
        "device": arguments.device,
    }
    refuse_other_conditions(out / "conditions.json", conditions)
    # Several lanes start together: the file is written once, so a reader never meets it empty.
    if not (out / "conditions.json").exists():
        (out / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
    print(json.dumps(conditions), flush=True)
    encoders = conditions["encoders"]
    if arguments.section == SELECT:
        plans = lane_of(
            select_plans(
                arguments.rates,
                arguments.lengths,
                encoders,
                arguments.seeds,
                events=sides.events_taught,
            ),
            arguments.lane,
        )
        _measure_fine_tuning(
            bench,
            out,
            plans,
            lane=arguments.lane,
            teach=sides.teach,
            taught=sides.taught,
            reads={SELECTION_SUFFIX: sides.read, "": bench.read},
        )
    elif arguments.section == BARS:
        bars = out / BARS_DIRECTORY
        (bars / "answers").mkdir(parents=True, exist_ok=True)
        measure_frozen(
            bench,
            bars,
            arguments.seeds,
            arm=f"{SELECT}-{DIRECTION}",
            events=sides.events_taught,
            fitting=[w.window for w in sides.teach],
            read=sides.read,
            taught=sides.taught,
            trees=False,
        )
    else:
        taught = tuple(w for w in bench.pool if AnomalyScheme.is_event(w.target))
        _measure_fine_tuning(
            bench,
            out,
            lane_of(refit_plans(out, arguments.seeds, events=every_event), arguments.lane),
            lane=arguments.lane,
            teach=bench.pool,
            taught=taught,
            reads={"": bench.read},
        )


def adopt(directory: Path, source: Path, arms: Mapping[str, str]) -> int:
    """Copy the source's cells of ``arms`` in, renamed, as bars on the validation months: how many.

    Raises:
        SystemExit: If an arm has no cell in the source, or a cell scored other windows than the
            fine-tuning cells here, which would pair nothing.
    """
    measured = sorted(
        path
        for path in (directory / "answers").glob("*.csv")
        if not path.name.endswith(f"{SELECTION_SUFFIX}.csv")
    )
    expected = _windows_of(read_answers(measured[0])) if measured else None
    (directory / "answers").mkdir(exist_ok=True)
    table_path = lane_table(directory, None)
    done = done_cells(directory)
    count = 0
    for arm, renamed in arms.items():
        cells = [cell for cell in cells_of(source) if cell["arm"] == arm]
        if not cells:
            raise SystemExit(f"{source} holds no cell of {arm}")
        for cell in cells:
            answers = read_answers(source / "answers" / f"{cell['cell']}.csv")
            if expected is not None and _windows_of(answers) != expected:
                raise SystemExit(f"{cell['cell']} scored other windows")
            name = cell_name(renamed, cell["backbone"], int(cell["events"]), int(cell["seed"]))
            if name in done:
                continue
            write_answers(directory / "answers" / f"{name}.csv", answers)
            record(
                table_path,
                (
                    name,
                    renamed,
                    cell["backbone"],
                    cell["events"],
                    cell["seed"],
                    "",
                    0,
                    0.0,
                    "",
                    "",
                    "",
                ),
            )
            count += 1
    return count


def _windows_of(answers: Iterable[WindowPrediction]) -> set[tuple[str, int, float]]:
    return {(str(each.window.unit), each.window.position, each.target) for each in answers}


@dataclass(frozen=True, kw_only=True)
class ReferenceRow:
    """One encoder: fine-tuning refitted on every event against the direction taught the same.

    Attributes:
        encoder: Which encoder both start from.
        areas: The refit's area under each seed.
        recall: The refit's share of events found at the stated false alarms, over the seeds.
        bar: The direction's mean area.
        difference: The refit's mean less the direction's.
        low: The interval's lower end, over months.
        high: The interval's upper end.
        p_value: The bootstrap's p-value.
        threshold: Twice the standard error over seeds: neither side was pretrained, or both
            were by the same run, so no spread between pretrainings enters.
    """

    encoder: str
    areas: Mapping[int, float]
    recall: float
    bar: float
    difference: float
    low: float
    high: float
    p_value: float
    threshold: float

    @property
    def outcome(self) -> str:
        """What the reading says, by the declared table."""
        if self.difference >= max(LEAST_GAIN, self.threshold) and self.low > 0.0:
            return "clearly above"
        if self.difference < -self.threshold and self.high < 0.0:
            return "below"
        return "around"


def reference_row(
    encoder: str,
    refit: Mapping[int, AnomalyRanking],
    direction: Mapping[int, AnomalyRanking],
    *,
    bootstrap: PairedUnitBootstrap,
) -> ReferenceRow:
    """The row of one encoder, seeds paired.

    Raises:
        SystemExit: If the direction lacks a seed the refit holds.
    """
    seeds = sorted(refit)
    if not set(seeds) <= set(direction):
        raise SystemExit(f"{encoder}: the direction lacks a seed the refit holds")
    differences = [
        refit[seed].average_precision - direction[seed].average_precision for seed in seeds
    ]
    compared = bootstrap.compare(
        PairedAnomalyRankings(
            control=tuple(direction[seed] for seed in seeds),
            candidate=tuple(refit[seed] for seed in seeds),
        )
    )
    standard_error = stdev(differences) / len(differences) ** 0.5 if len(differences) > 1 else 0.0
    return ReferenceRow(
        encoder=encoder,
        areas={seed: refit[seed].average_precision for seed in seeds},
        recall=fmean(refit[seed].event_recall_at_false_alarm_share(SHARE) for seed in seeds),
        bar=fmean(direction[seed].average_precision for seed in seeds),
        difference=fmean(differences),
        low=compared.interval.low,
        high=compared.interval.high,
        p_value=compared.p_value,
        threshold=DetectionThreshold(value=2.0 * standard_error).value,
    )


def reference_table(rows: Sequence[ReferenceRow]) -> str:
    return table(
        (
            "Encoder",
            "Refit per seed",
            "Refit",
            "Recall @1 %",
            "Direction",
            "Difference [interval]",
            "p",
            "T",
            "Outcome",
        ),
        [
            (
                row.encoder,
                " ".join(f"{area:.3f}" for area in row.areas.values()),
                f"{fmean(row.areas.values()):.3f}",
                f"{row.recall:.3f}",
                f"{row.bar:.3f}",
                f"{row.difference:+.3f} [{row.low:+.3f}; {row.high:+.3f}]",
                f"{row.p_value:.3f}",
                f"{row.threshold:.3f}",
                row.outcome,
            )
            for row in rows
        ],
    )


def rankings_of(
    directory: Path, cells: Iterable[Mapping[str, str]]
) -> dict[str, dict[str, dict[int, AnomalyRanking]]]:
    """Every cell's ranking on the validation months, by arm, encoder and seed."""
    found: dict[str, dict[str, dict[int, AnomalyRanking]]] = defaultdict(lambda: defaultdict(dict))
    for cell in cells:
        answers = read_answers(directory / "answers" / f"{cell['cell']}.csv")
        found[cell["arm"]][cell["backbone"]][int(cell["seed"])] = AnomalyRanking.of(answers)
    return found


def missing_refits(directory: Path, cells: Iterable[Mapping[str, str]]) -> list[str]:
    """The refit cells ``chosen.json`` asks for that are not in."""
    conditions = json.loads((directory / "conditions.json").read_text())
    held = {cell["cell"] for cell in cells}
    plans = refit_plans(directory, conditions["seeds"], events=conditions["events_in_pool"])
    return sorted(plan.cell for plan in plans if plan.cell not in held)


def read_tables(directory: Path, *, partial: bool = False) -> str:
    """The reference against its bars, then the curve on the validation months as a description.

    Raises:
        SystemExit: If a refit cell is missing and ``partial`` is not asked, or no cell is in.
    """
    cells = cells_of(directory)
    if not cells:
        raise SystemExit(f"no cell in {directory}")
    if not partial:
        missing = missing_refits(directory, cells)
        if missing:
            raise SystemExit(f"{len(missing)} refit cells are missing, the first {missing[0]}")
    found = rankings_of(directory, cells)
    bootstrap = PairedUnitBootstrap()
    direction = found.get(DIRECTION, {})
    refits: dict[str, dict[int, AnomalyRanking]] = {}
    for arm, sides in found.items():
        if not arm.startswith(REFIT):
            continue
        for encoder, side in sides.items():
            if encoder in refits:
                raise SystemExit(f"{encoder} holds two refit arms: a grid that grew left the old")
            refits[encoder] = side
    if not partial:
        without = sorted(encoder for encoder in refits if encoder not in direction)
        if without:
            raise SystemExit(f"adopt the bars first: no direction under {without}")
    rows = [
        reference_row(encoder, refits[encoder], direction[encoder], bootstrap=bootstrap)
        for encoder in sorted(refits)
        if encoder in direction
    ]
    areas = table(
        ("Arm", "Encoder", "Area per seed", "Area", "Recall @1 %"),
        [
            (
                arm,
                encoder,
                " ".join(f"{side[seed].average_precision:.3f}" for seed in sorted(side)),
                f"{fmean(each.average_precision for each in side.values()):.3f}",
                f"{fmean(e.event_recall_at_false_alarm_share(SHARE) for e in side.values()):.3f}",
            )
            for arm, sides in sorted(found.items())
            if not arm.startswith(SELECT)
            for encoder, side in sorted(sides.items())
        ],
    )
    gains = []
    for arm, sides in sorted(found.items()):
        if arm.startswith(SELECT) or INITIALISATION not in sides or len(sides) < 2:
            continue
        base = sides[INITIALISATION]
        for name in sorted(encoder for encoder in sides if encoder != INITIALISATION):
            seeds = sorted(base)
            compared = bootstrap.compare(
                PairedAnomalyRankings(
                    control=tuple(base[seed] for seed in seeds),
                    candidate=tuple(sides[name][seed] for seed in seeds),
                )
            )
            gains.append(
                (
                    arm,
                    name,
                    f"{compared.reduction:+.3f} [{compared.interval.low:+.3f}; "
                    f"{compared.interval.high:+.3f}]",
                )
            )
    curve_rows = []
    for arm, sides in sorted(found.items()):
        if not arm.startswith(SELECT):
            continue
        _, rate, steps = split_arm(arm)
        curve_rows += [
            (
                f"{rate:g}",
                str(steps),
                encoder,
                f"{fmean(each.average_precision for each in side.values()):.3f}",
                " ".join(f"{side[seed].average_precision:.3f}" for seed in sorted(side)),
            )
            for encoder, side in sorted(sides.items())
        ]
    curve = table(
        ("Rate", "Steps", "Encoder", "Area on validation months", "Per seed"),
        sorted(curve_rows, key=lambda row: (float(row[0]), int(row[1]), row[2])),
    )
    mechanics = table(
        ("Cell", "Taught windows", "Steps", "Seconds", "Loss early", "Loss late", "Moved"),
        [
            (
                cell["cell"],
                cell["taught_windows"],
                cell["steps"],
                f"{float(cell['seconds']):.0f}",
                cell["first_loss"][:6],
                cell["last_loss"][:6],
                cell["moved"][:6],
            )
            for cell in cells
            if cell["last_loss"]
        ],
    )
    text = "\n\n".join(
        (
            reference_table(rows),
            areas,
            table(("Arm", "Encoder", "Pretrained less initialisation"), gains),
            curve,
        )
    )
    (directory / "reference.md").write_text(text + "\n")
    (directory / "mechanics.md").write_text(mechanics + "\n")
    return text


def split_by_seen(
    predictions: Iterable[WindowPrediction], classes: Mapping[int, str], seen: frozenset[str]
) -> dict[str, list[WindowPrediction]]:
    """Two halves: the normal windows with the events of seen classes, and with those of unseen."""
    split: dict[str, list[WindowPrediction]] = {SEEN_HALF: [], UNSEEN_HALF: []}
    for each in predictions:
        if each.target == AnomalyScheme.NORMAL:
            split[SEEN_HALF].append(each)
            split[UNSEEN_HALF].append(each)
        else:
            split[SEEN_HALF if classes[int(each.target)] in seen else UNSEEN_HALF].append(each)
    return split


def class_rows(
    cells: Iterable[Mapping[str, str]],
    directory: Path,
    classes: Mapping[int, str],
    seen: frozenset[str],
) -> list[tuple[object, ...]]:
    """One row per refit or bar cell and half; a half holding no event is left out."""
    rows: list[tuple[object, ...]] = []
    for cell in cells:
        if cell["arm"].startswith(SELECT):
            continue
        answers = read_answers(directory / "answers" / f"{cell['cell']}.csv")
        for half, predictions in split_by_seen(answers, classes, seen).items():
            scored_events = {int(each.target) for each in predictions if each.target}
            if not scored_events:
                continue
            ranking = AnomalyRanking.of(predictions)
            rows.append(
                (
                    cell["arm"],
                    cell["backbone"],
                    int(cell["seed"]),
                    half,
                    len(scored_events),
                    ranking.average_precision,
                    ranking.event_recall_at_false_alarm_share(SHARE),
                )
            )
    return rows


def class_table(rows: Iterable[Sequence[object]]) -> str:
    """Each arm's area on the events of seen classes and of unseen ones, over the seeds."""
    areas: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    events: dict[str, int] = {}
    for row in rows:
        areas[(str(row[0]), str(row[1]))][str(row[3])].append(float(str(row[5])))
        events[str(row[3])] = int(str(row[4]))
    return table(
        (
            "Arm",
            "Encoder",
            f"Seen classes ({events.get(SEEN_HALF, 0)} events)",
            f"Unseen classes ({events.get(UNSEEN_HALF, 0)} events)",
        ),
        [
            (
                arm,
                encoder,
                f"{fmean(halves[SEEN_HALF]):.3f}" if halves[SEEN_HALF] else "",
                f"{fmean(halves[UNSEEN_HALF]):.3f}" if halves[UNSEEN_HALF] else "",
            )
            for (arm, encoder), halves in sorted(areas.items())
        ],
    )


def classes(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Read the refit and the bars by whether an event's class was among those taught."""
    from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS, EsaAdRawCorpusReader
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
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
    seen = frozenset(fitting.classes[int(w.target)] for w in fitting.pool)
    cells = cells_of(arguments.dir)
    missing = missing_refits(arguments.dir, cells)
    if missing and not arguments.partial:
        raise SystemExit(f"{len(missing)} refit cells are missing, the first {missing[0]}")
    rows = class_rows(cells, arguments.dir, fitting.classes, seen)
    with (arguments.dir / "classes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CLASS_COLUMNS)
        writer.writerows(rows)
    text = class_table(rows)
    (arguments.dir / "classes.md").write_text(text + "\n")
    print(text)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True)
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument("--weights", nargs=3, action="append", required=True)
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--rates", type=float, nargs="+", required=True)
    measuring.add_argument("--lengths", type=int, nargs="+", required=True)
    measuring.add_argument(
        "--select-from", required=True, help="the first month that chooses, as YYYY-MM"
    )
    measuring.add_argument("--section", choices=SECTIONS, default=SELECT)
    measuring.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    measuring.add_argument("--lane", help="i/n: this process takes every n-th cell from the i-th")
    measuring.add_argument("--device", default="mps")
    measuring.add_argument(
        "--trial", action="store_true", help="run before the declaration is committed"
    )
    choosing = commands.add_parser("choose")
    choosing.add_argument("--dir", type=Path, required=True)
    choosing.add_argument(
        "--partial", action="store_true", help="choose from a grid not yet whole, as a trial does"
    )
    adopting = commands.add_parser("adopt")
    adopting.add_argument("--dir", type=Path, required=True)
    adopting.add_argument("--from", dest="source", type=Path, required=True)
    adopting.add_argument(
        "--arm", nargs=2, action="append", required=True, metavar=("THEIRS", "OURS")
    )
    by_class = commands.add_parser("classes")
    by_class.add_argument("--task", required=True)
    by_class.add_argument("--dir", type=Path, required=True)
    by_class.add_argument("--first", required=True)
    by_class.add_argument("--last", required=True)
    by_class.add_argument("--partial", action="store_true", help="read before the refit is whole")
    reading = commands.add_parser("read")
    reading.add_argument("--dir", type=Path, required=True)
    reading.add_argument(
        "--partial", action="store_true", help="read before every refit cell is in, as a trial does"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure a section, choose, adopt the bars, read by class, or read the reference."""
    arguments = parse_arguments(argv)
    if arguments.command == "measure":  # pragma: no cover - environment
        measure(arguments)
    elif arguments.command == "choose":
        print(choice_table(choose_from(arguments.dir, partial=arguments.partial)))
    elif arguments.command == "adopt":
        count = adopt(arguments.dir, arguments.source, dict(arguments.arm))
        print(f"{count} cells adopted")
    elif arguments.command == "classes":  # pragma: no cover - environment
        classes(arguments)
    else:
        print(read_tables(arguments.dir, partial=arguments.partial))


if __name__ == "__main__":  # pragma: no cover
    main()
