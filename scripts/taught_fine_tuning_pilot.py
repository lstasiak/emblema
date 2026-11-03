"""Whether fine-tuning on a few taught events gains much more under a pretrained encoder.

A detection task's frozen states match their initialisation with or without taught events. The one
route those readings cannot speak for is the encoder learning from the events. This pilot asks only
whether that gain is large. The events of the fitting side teach and the validation months are read,
which the pretrained encoders never read, so the encoders differ in what they learnt and not in what
they have seen. Each arm starts from one encoder and learns a head and the encoder by one rule: half
of a batch are windows of the taught events, an event drawn first so that each weighs alike, and
half are the fitting side's other windows, counted as normal. Under one seed both encoders see the
same events and the same batches; the task's registered identity draws the events, so they are the
events the same seed taught a campaign's detectors.

``measure`` touches torch and writes ``cells.csv`` (conditions, cost and the training's mechanics,
no area) and ``answers/<cell>.csv`` (every read window's score). A cell already answered is skipped,
so a cut run resumes. ``adopt`` copies a campaign's detectors in from its export as bars, refusing
any that scored other windows. ``read`` computes the tables from those files and nothing else, so a
trial of ``measure`` stays blind.

    uv run --env-file .env.r2 scripts/taught_fine_tuning_pilot.py measure
        --task esa-mission1-events --task-id UUID --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --weights base2 KEY CHECKSUM --out DIR
        --rates 0.001 0.0001 [--events 5 10] [--seeds 1 2 3] [--device mps]
    uv run scripts/taught_fine_tuning_pilot.py adopt --predictions PREDICTIONS.csv --dir DIR
        --bar direction base1 CAMPAIGN CANDIDATE EVENTS [--bar ...] [--seeds 1 2 3]
    uv run scripts/taught_fine_tuning_pilot.py read --dir DIR

``measure`` reads the registered schedule from ``EMBLEMA_WORKER__SCHEDULE__*``.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any
from uuid import UUID

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.detection.taught_events import TaughtEvents
from emblema.evaluation.domain.exceptions import DivergedAdaptationError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_reading import PairedReading
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.pretraining_spread import PretrainingSpread
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from scripts.reporting import table

SHARE = 0.01
# The floor the register reads a pretraining's spread with.
SPREAD_FLOOR = 0.014
# The least gain a first-mission campaign counts, registered with its rules.
LEAST_GAIN = 0.05
CELL_COLUMNS = (
    "cell",
    "arm",
    "backbone",
    "events",
    "seed",
    "taught_windows",
    "steps",
    "seconds",
    "first_loss",
    "last_loss",
    "moved",
)
ANSWER_COLUMNS = ("unit", "position", "ends_at", "target", "predicted")
INITIALISATION = "initialisation"
FINE_TUNED = "ft@"


class BalancedDraw:
    """Which windows each step learns from: half of a taught event, half of the others.

    An event is drawn first and then one of its windows, so an event of a month and an event of an
    hour weigh alike, as the task's measure weighs them. Seeded, so both arms of a seed see the same
    batches.
    """

    def __init__(
        self, events: Mapping[int, Sequence[int]], others: Sequence[int], *, seed: int
    ) -> None:
        if not events or not others:
            raise SystemExit("a step needs a taught window and another")
        self._events = [list(events[event]) for event in sorted(events)]
        self._others = list(others)
        self._draws = random.Random(seed)

    def batch(self, size: int) -> tuple[list[int], list[float]]:
        """Indices of ``size`` windows and whether each is an event's."""
        taught = size // 2
        indices = [self._draws.choice(self._draws.choice(self._events)) for _ in range(taught)] + [
            self._draws.choice(self._others) for _ in range(size - taught)
        ]
        return indices, [1.0] * taught + [0.0] * (size - taught)


def fine_tuned(
    model: nn.Module,
    answer: Callable[[Sequence[int]], torch.Tensor],
    draw: BalancedDraw,
    schedule: AdaptationSchedule,
    *,
    device: str,
) -> tuple[float, float]:
    """Run the schedule's floor of steps on balanced batches: the loss early and late.

    The optimiser, the shape of the rate and the refusal of a loss that stops being finite are the
    harness's own; only the batches differ, since a few events among tens of thousands of windows
    would hardly reach a batch drawn in order.

    Raises:
        DivergedAdaptationError: If a batch's loss stops being finite.
    """
    steps = schedule.min_steps
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(
        trainable, lr=schedule.learning_rate, weight_decay=schedule.weight_decay
    )
    rate = torch.optim.lr_scheduler.LambdaLR(
        optimiser, schedule.learning_rate_schedule(steps * schedule.batch_size, 1).factor
    )
    loss = nn.BCEWithLogitsLoss()
    seen: list[float] = []
    model.train()
    for step in range(steps):
        indices, targets = draw.batch(schedule.batch_size)
        batch_loss = loss(
            answer(indices), torch.tensor(targets, dtype=torch.float32, device=device)
        )
        value = float(batch_loss.detach())
        if not np.isfinite(value):
            raise DivergedAdaptationError(f"the loss of step {step} is {value}")
        optimiser.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimiser.step()
        rate.step()
        seen.append(value)
    tenth = max(1, steps // 10)
    return fmean(seen[:tenth]), fmean(seen[-tenth:])


def moved_share(before: Mapping[str, torch.Tensor], model: nn.Module) -> float:
    """How far the weights travelled: the norm of the change over the norm of the start."""
    change = start = 0.0
    for name, parameter in model.named_parameters():
        if name in before:
            first = before[name].to(parameter.device)
            change += float((parameter.detach() - first).pow(2).sum())
            start += float(first.pow(2).sum())
    return (change / start) ** 0.5 if start else 0.0


def read_answers(path: Path) -> list[WindowPrediction]:
    with path.open(newline="") as handle:
        return [
            WindowPrediction(
                window=TaskWindow(
                    unit=UnitKey(row["unit"]),
                    position=int(row["position"]),
                    ends_at=float(row["ends_at"]),
                ),
                target=float(row["target"]),
                predicted=float(row["predicted"]),
            )
            for row in csv.DictReader(handle)
        ]


def write_answers(path: Path, predictions: Iterable[WindowPrediction]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANSWER_COLUMNS)
        for each in predictions:
            writer.writerow(
                (
                    each.window.unit,
                    each.window.position,
                    each.window.ends_at,
                    each.target,
                    each.predicted,
                )
            )


def cell_name(arm: str, backbone: str, events: int, seed: int) -> str:
    return f"{arm}-{backbone}-k{events}-s{seed}"


@dataclass(frozen=True, kw_only=True)
class GainRow:
    """Fine-tuning over a pretrained encoder less the same over its initialisation, at one count.

    Attributes:
        arm: The arm read.
        events: How many events were taught.
        pretrained: The arm's area under the pretrained encoders, each by name.
        initialisation: The arm's area at the initialisation.
        gain: The mean of the pretrained encoders' gains.
        low: The interval's lower end, over months, on the pooled pair.
        high: The interval's upper end.
        p_value: The bootstrap's p-value on the pooled pair.
        threshold: What the gain must pass by the registered rule.
        rejected: Whether the p-value stands under Holm's correction over the fine-tuning rows.
        recall_pretrained: Share of events found at the stated false alarms, pretrained.
        recall_initialisation: The same at the initialisation.
    """

    arm: str
    events: int
    pretrained: Mapping[str, float]
    initialisation: float
    gain: float
    low: float
    high: float
    p_value: float
    threshold: float
    rejected: bool
    recall_pretrained: float
    recall_initialisation: float

    @property
    def harms(self) -> bool:
        """A loss past the threshold, with an interval below zero."""
        return self.gain < -self.threshold and self.high < 0.0

    @property
    def rules_out_the_least_gain(self) -> bool:
        """Whether the least gain counted lies above both what pretraining and months allow."""
        return max(self.gain + self.threshold, self.high) < LEAST_GAIN

    @property
    def outcome(self) -> str:
        """What the reading says: only a bound below the least gain closes the question."""
        if self.passes:
            return "large gain"
        if self.harms:
            return "harm"
        if self.rules_out_the_least_gain:
            return "closed"
        return "inconclusive"

    @property
    def passes(self) -> bool:
        """Past the threshold and the least gain counted, standing under the correction."""
        return self.rejected and self.gain >= max(self.threshold, LEAST_GAIN) and self.low > 0.0


def between_pretrainings(
    first: Mapping[int, AnomalyRanking],
    second: Mapping[int, AnomalyRanking],
    seeds: Sequence[int],
    *,
    bootstrap: PairedUnitBootstrap,
) -> PairedReading:
    """The difference between two pretrainings, seeds paired, as the spread's rule reads it."""
    differences = [first[seed].average_precision - second[seed].average_precision for seed in seeds]
    compared = bootstrap.compare(
        PairedAnomalyRankings(
            control=tuple(second[seed] for seed in seeds),
            candidate=tuple(first[seed] for seed in seeds),
        )
    )
    return PairedReading(
        mean=fmean(differences),
        standard_error=stdev(differences) / len(differences) ** 0.5
        if len(differences) > 1
        else 0.0,
        interval=compared.interval,
    )


def spread_of(
    readings: Iterable[Mapping[str, Mapping[int, AnomalyRanking]]],
    *,
    bootstrap: PairedUnitBootstrap,
) -> float:
    """The one spread between pretrainings an arm's readings are judged by, over all of them.

    One pair of pretrainings at one reading says very little: where they happen to agree, its own
    spread would drop to the floor and lower that reading's bar alone.
    """
    pairs = [
        between_pretrainings(sides[first], sides[second], sorted(sides[first]), bootstrap=bootstrap)
        for sides in readings
        for at, first in enumerate(sorted(sides))
        for second in sorted(sides)[at + 1 :]
    ]
    return PretrainingSpread.of(pairs, floor=SPREAD_FLOOR).value if pairs else SPREAD_FLOOR


def gain_row(
    arm: str,
    events: int,
    pretrained: Mapping[str, Mapping[int, AnomalyRanking]],
    initialisation: Mapping[int, AnomalyRanking],
    *,
    spread: float,
    bootstrap: PairedUnitBootstrap,
) -> GainRow:
    """The row of one arm at one count, the seeds paired and the pretrainings pooled.

    Raises:
        SystemExit: If the pretrained encoders and the initialisation hold different seeds.
    """
    seeds = sorted(initialisation)
    names = sorted(pretrained)
    if any(sorted(pretrained[name]) != seeds for name in names):
        raise SystemExit(f"{arm} at {events} events does not hold the same seeds on every side")
    base = {seed: initialisation[seed].average_precision for seed in seeds}
    gains = {
        name: {seed: pretrained[name][seed].average_precision - base[seed] for seed in seeds}
        for name in names
    }
    per_seed = [fmean(gains[name][seed] for name in names) for seed in seeds]
    standard_error = stdev(per_seed) / len(per_seed) ** 0.5 if len(per_seed) > 1 else 0.0
    threshold = DetectionThreshold.of_gain(
        standard_error=standard_error, spread=spread, pretrainings=len(names)
    )
    pooled = PairedAnomalyRankings(
        control=tuple(initialisation[seed] for seed in seeds) * len(names),
        candidate=tuple(pretrained[name][seed] for name in names for seed in seeds),
    )
    compared = bootstrap.compare(pooled)
    recalls = {
        name: fmean(
            pretrained[name][seed].event_recall_at_false_alarm_share(SHARE) for seed in seeds
        )
        for name in names
    }
    return GainRow(
        arm=arm,
        events=events,
        pretrained={
            name: fmean(pretrained[name][seed].average_precision for seed in seeds)
            for name in names
        },
        initialisation=fmean(base.values()),
        gain=fmean(per_seed),
        low=compared.interval.low,
        high=compared.interval.high,
        p_value=compared.p_value,
        threshold=threshold.value,
        rejected=compared.p_value < 0.05,
        recall_pretrained=fmean(recalls.values()),
        recall_initialisation=fmean(
            initialisation[seed].event_recall_at_false_alarm_share(SHARE) for seed in seeds
        ),
    )


def render(rows: Sequence[GainRow]) -> str:
    return table(
        (
            "Arm",
            "Events",
            "Pretrained",
            "Initialisation",
            "Gain [interval]",
            "p",
            "T",
            "Recall @1 %",
            "Outcome",
        ),
        [
            (
                row.arm,
                str(row.events),
                ", ".join(f"{name} {area:.3f}" for name, area in sorted(row.pretrained.items())),
                f"{row.initialisation:.3f}",
                f"{row.gain:+.3f} [{row.low:+.3f}; {row.high:+.3f}]",
                f"{row.p_value:.3f}",
                f"{row.threshold:.3f}",
                f"{row.recall_pretrained:.3f} / {row.recall_initialisation:.3f}",
                row.outcome,
            )
            for row in rows
        ],
    )


def areas_table(
    rankings: Mapping[tuple[str, int], Mapping[str, Mapping[int, AnomalyRanking]]],
) -> str:
    """Every arm's area and recall under every encoder, the bars beside fine-tuning."""
    return table(
        ("Arm", "Events", "Encoder", "Area", "Recall @1 %", "Seeds"),
        [
            (
                arm,
                str(events),
                name,
                f"{fmean(each.average_precision for each in side.values()):.3f}",
                f"{fmean(e.event_recall_at_false_alarm_share(SHARE) for e in side.values()):.3f}",
                str(len(side)),
            )
            for (arm, events), sides in sorted(rankings.items())
            for name, side in sorted(sides.items())
        ],
    )


def corrected(rows: Sequence[GainRow], *, family_size: int) -> list[GainRow]:
    """The rows with fine-tuning's readings, one family, judged under Holm's correction.

    A reading missing from the family counts against the rest. The bars are read beside the family
    and pass nothing.
    """
    family = [row for row in rows if row.arm.startswith(FINE_TUNED)]
    if not family:
        return list(rows)
    standing = HolmCorrection().rejected([row.p_value for row in family], family_size=family_size)
    judged = dict(zip((id(row) for row in family), standing, strict=True))
    return [replace(row, rejected=judged.get(id(row), False)) for row in rows]


def missing_cells(conditions: Mapping[str, Any], held: Iterable[str]) -> list[str]:
    """The fine-tuning cells the declared grid holds and the table does not."""
    encoders = [*conditions["encoders"], INITIALISATION]
    expected = {
        cell_name(f"{FINE_TUNED}{rate}", encoder, events, seed)
        for rate in conditions["rates"]
        for encoder in encoders
        for events in conditions["events"]
        for seed in conditions["seeds"]
    }
    return sorted(expected - set(held))


def read(directory: Path, *, partial: bool = False) -> str:
    """The tables, from the files ``measure`` wrote.

    Args:
        directory: Where ``measure`` wrote.
        partial: Read a grid not yet whole, as a trial does; the family stays its declared size.

    Raises:
        SystemExit: If no cell was measured, or the grid is not whole and ``partial`` is not asked.
    """
    with (directory / "cells.csv").open(newline="") as handle:
        cells = list(csv.DictReader(handle))
    if not cells:
        raise SystemExit(f"no cell in {directory}")
    conditions = json.loads((directory / "conditions.json").read_text())
    missing = missing_cells(conditions, (cell["cell"] for cell in cells))
    if missing and not partial:
        raise SystemExit(f"{len(missing)} declared cells are missing, the first {missing[0]}")
    rankings: dict[tuple[str, int], dict[str, dict[int, AnomalyRanking]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for cell in cells:
        answers = read_answers(directory / "answers" / f"{cell['cell']}.csv")
        rankings[(cell["arm"], int(cell["events"]))][cell["backbone"]][int(cell["seed"])] = (
            AnomalyRanking.of(answers)
        )
    bootstrap = PairedUnitBootstrap()
    pretrained_of = {
        key: {name: side for name, side in sides.items() if name != INITIALISATION}
        for key, sides in rankings.items()
        if INITIALISATION in sides and len(sides) > 1
    }
    # Fine-tuning's rates are one family; a bar's readings are judged by their own spread.
    spreads = {
        family: spread_of(
            [
                sides
                for (arm, _), sides in pretrained_of.items()
                if arm.startswith(FINE_TUNED) == family
            ],
            bootstrap=bootstrap,
        )
        for family in (True, False)
    }
    rows = [
        gain_row(
            arm,
            events,
            pretrained,
            rankings[(arm, events)][INITIALISATION],
            spread=spreads[arm.startswith(FINE_TUNED)],
            bootstrap=bootstrap,
        )
        for (arm, events), pretrained in sorted(pretrained_of.items())
    ]
    rows = corrected(rows, family_size=len(conditions["rates"]) * len(conditions["events"]))
    (directory / "gains.md").write_text(render(rows) + "\n")
    (directory / "areas.md").write_text(areas_table(rankings) + "\n")
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
        ],
    )
    (directory / "mechanics.md").write_text(mechanics + "\n")
    return render(rows)


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every fine-tuning cell of the pilot, on the device, those already answered skipped."""
    from emblema.config.settings import Settings
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.configured import configured_store
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from emblema.entrypoints.restored_backbones import RestoredBackbones
    from emblema.entrypoints.workers.declared_worker import DeclaredWorker
    from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
    from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
        InMemoryDownstreamTaskRepository,
    )
    from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
    from emblema.evaluation.application.use_cases.define_downstream_task import (
        DefineDownstreamTask,
    )
    from emblema.evaluation.contracts.identifiers import TaskId
    from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
    from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
    from emblema.shared.adapters.tensors.token_tensors import TokenTensors
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum

    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = committed_tree()
    refuse_another_commit(out / "conditions.json", commit)
    settings = Settings()
    store = configured_store(settings)
    worker = settings.require_worker()
    blocks = PublishedCorpusBlocks(store, worker.workspace)
    corpus = BlockCorpusWindows(blocks)
    manifest_ref = ArtifactRef(arguments.manifest[0], Checksum.parse(arguments.manifest[1]))
    tasks = InMemoryDownstreamTaskRepository()
    known = KnownTasks.named(arguments.task)
    defined = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        known.defined_over(manifest_ref, corpus.describe(manifest_ref))
    )
    task: DownstreamTask = tasks.get(defined)
    # The draw of the events ranks by the task's identity, so the registered one draws the same.
    registered = TaskId(UUID(arguments.task_id))
    truth = KnownGroundTruths.under(Path("data/raw"))
    fitting = task.read_windows(corpus.windows_of(task.manifest, task.tuning_units))
    pool = task.labelled(fitting, truth.truths_of(task.corpus, fitting))
    validation = task.read_windows(corpus.windows_of(task.manifest, task.validation_units))
    read = task.labelled(validation, truth.truths_of(task.corpus, validation))
    manifest = blocks.manifest_of(task.manifest)
    block = blocks.block_of(manifest)
    schedule = DeclaredWorker(worker).schedule()
    weights = {
        name: ArtifactRef(key, Checksum.parse(checksum))
        for name, key, checksum in arguments.weights
    }
    backbones = {name: RestoredBackbones(store, ref) for name, ref in weights.items()}
    first = next(iter(backbones.values()))
    layout = BinnedTokens(first.bins)
    laid_pool = layout.lay(block.at([w.window.position for w in pool]))
    laid_read = layout.lay(block.at([w.window.position for w in read]))
    units = [str(w.window.unit) for w in read]
    conditions = {
        "commit": commit,
        "encoders": list(weights),
        "events": arguments.events,
        "seeds": arguments.seeds,
        "windows_taught_from": len(pool),
        "windows_read": len(read),
        "months_read": len(set(units)),
        "events_in_pool": len({int(w.target) for w in pool if w.target}),
        "events_read": len({int(w.target) for w in read if w.target}),
        "rates": arguments.rates,
        "steps": schedule.min_steps,
        "batch": schedule.batch_size,
        "device": arguments.device,
    }
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
    print(json.dumps(conditions))
    table_path = out / "cells.csv"
    done = done_cells(table_path)
    position_of = {w.window.position: at for at, w in enumerate(pool)}

    def answering(
        candidate: nn.Module, laid: Sequence[LaidWindow]
    ) -> Callable[[Sequence[int]], torch.Tensor]:
        def answer(indices: Sequence[int]) -> torch.Tensor:
            batch = TokenTensors.of(layout.batch([laid[index] for index in indices]))
            answers: torch.Tensor = candidate(batch.to(arguments.device))
            return answers.squeeze(-1)

        return answer

    arms = [*backbones.items(), (INITIALISATION, first)]
    for events in arguments.events:
        for seed in arguments.seeds:
            taught = TaughtEvents.drawn(registered, pool, events=events, seed=seed).windows
            spans: dict[int, list[int]] = defaultdict(list)
            for window in taught:
                spans[int(window.target)].append(position_of[window.window.position])
            taught_at = {index for span in spans.values() for index in span}
            others = [index for index in range(len(pool)) if index not in taught_at]
            for name, backbone in arms:
                for learning_rate in arguments.rates:
                    fine_tuning = f"{FINE_TUNED}{learning_rate}"
                    cell = cell_name(fine_tuning, name, events, seed)
                    if cell in done:
                        continue
                    rated = replace(schedule, learning_rate=learning_rate)
                    started = time.perf_counter()
                    torch.manual_seed(seed)
                    plan = AdaptationPlan(
                        mode=(
                            TransferMode.FROM_SCRATCH
                            if name == INITIALISATION
                            else TransferMode.FULL_FINE_TUNING
                        ),
                        backbone=weights.get(name),
                        schedule=rated,
                        lora=None,
                        seed=seed,
                    )
                    candidate = AdaptedBackbone.under(
                        plan,
                        backbone,
                        vocabulary_size=len(manifest.channels),
                        starting_at=0.0,
                        layout=layout,
                    ).to(arguments.device)
                    before = {
                        key: value.detach().clone() for key, value in candidate.named_parameters()
                    }
                    early, late = fine_tuned(
                        candidate,
                        answering(candidate, laid_pool),
                        BalancedDraw(spans, others, seed=seed),
                        rated,
                        device=arguments.device,
                    )
                    moved = moved_share(before, candidate)
                    candidate.eval()
                    reading = answering(candidate, laid_read)
                    with torch.no_grad():
                        scores = torch.cat(
                            [
                                reading(range(at, min(at + 64, len(laid_read)))).to("cpu")
                                for at in range(0, len(laid_read), 64)
                            ]
                        )
                    raw: NDArray[np.float64] = scores.to(torch.float64).numpy()
                    alarm = ScoreAlarm.set_over(raw, contamination=SHARE, units=units)
                    write_answers(
                        out / "answers" / f"{cell}.csv",
                        (
                            WindowPrediction(
                                window=w.window, target=w.target, predicted=float(score)
                            )
                            for w, score in zip(
                                read, alarm.above(raw, units=units).tolist(), strict=True
                            )
                        ),
                    )
                    record(
                        table_path,
                        (
                            cell,
                            fine_tuning,
                            name,
                            events,
                            seed,
                            len(taught),
                            rated.min_steps,
                            time.perf_counter() - started,
                            early,
                            late,
                            moved,
                        ),
                    )
                    print(f"{cell}: {time.perf_counter() - started:.0f} s", flush=True)


def committed_tree() -> str:  # pragma: no cover - environment
    """The commit the tree stands at, which the cells then belong to.

    Raises:
        SystemExit: If a tracked file differs from it.
    """
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    if changed.strip():
        raise SystemExit("tracked files have changed: a cell belongs to one commit")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def refuse_another_commit(path: Path, commit: str) -> None:
    """Refuse to add cells to a run begun at another commit.

    Raises:
        SystemExit: If ``path`` names another commit.
    """
    if path.exists():
        begun = json.loads(path.read_text()).get("commit")
        if begun != commit:
            raise SystemExit(f"this run began at {begun}, the tree is at {commit}")


def done_cells(table_path: Path) -> set[str]:
    """The cells a table holds, the table begun if there is none."""
    if not table_path.exists():
        with table_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(CELL_COLUMNS)
        return set()
    with table_path.open(newline="") as handle:
        return {row["cell"] for row in csv.DictReader(handle)}


def record(table_path: Path, row: Sequence[object]) -> None:
    with table_path.open("a", newline="") as handle:
        csv.writer(handle).writerow(row)


@dataclass(frozen=True, kw_only=True)
class Bar:
    """A campaign's detector read as a bar beside fine-tuning.

    Attributes:
        arm: The arm it is read as.
        encoder: The encoder it stands for.
        campaign: The campaign that ran it.
        candidate: Its name in that campaign.
        events: How many events it was taught.
    """

    arm: str
    encoder: str
    campaign: str
    candidate: str
    events: int


def adopt(predictions: Path, directory: Path, bars: Sequence[Bar], seeds: Sequence[int]) -> int:
    """Copy each bar's answers at ``seeds`` in as cells, in one pass over the export: how many.

    Raises:
        SystemExit: If a bar's seed is absent, or it scored other windows than the cells measured
            here, which would pair nothing.
    """
    wanted = {(bar.campaign, bar.candidate): bar for bar in bars}
    held: dict[tuple[Bar, int], list[WindowPrediction]] = defaultdict(list)
    with predictions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            bar = wanted.get((row["campaign"], row["candidate"]))
            if bar is not None and int(row["seed"]) in seeds:
                held[(bar, int(row["seed"]))].append(
                    WindowPrediction(
                        window=TaskWindow(
                            unit=UnitKey(row["unit"]),
                            position=int(row["position"]),
                            ends_at=float(row["ends_at"]),
                        ),
                        target=float(row["target"]),
                        predicted=float(row["predicted"]),
                    )
                )
    table_path = directory / "cells.csv"
    done = done_cells(table_path)
    measured = sorted((directory / "answers").glob(f"{FINE_TUNED}*.csv"))
    expected = _windows_of(read_answers(measured[0])) if measured else None
    (directory / "answers").mkdir(exist_ok=True)
    count = 0
    for bar in bars:
        for seed in seeds:
            answers = held.get((bar, seed))
            if not answers:
                raise SystemExit(f"{bar.candidate} of {bar.campaign} holds no seed {seed}")
            if expected is not None and _windows_of(answers) != expected:
                raise SystemExit(f"{bar.candidate} of {bar.campaign} scored other windows")
            cell = cell_name(bar.arm, bar.encoder, bar.events, seed)
            if cell in done:
                continue
            write_answers(directory / "answers" / f"{cell}.csv", answers)
            record(
                table_path, (cell, bar.arm, bar.encoder, bar.events, seed, "", 0, 0.0, "", "", "")
            )
            count += 1
    return count


def _windows_of(answers: Iterable[WindowPrediction]) -> set[tuple[str, int, float]]:
    return {(str(each.window.unit), each.window.position, each.target) for each in answers}


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True, help="the task's identity in the registry")
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument("--weights", nargs=3, action="append", required=True)
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--rates", type=float, nargs="+", required=True)
    measuring.add_argument("--events", type=int, nargs="+", default=[5, 10])
    measuring.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    measuring.add_argument("--device", default="mps")
    adopting = commands.add_parser("adopt")
    adopting.add_argument("--predictions", type=Path, required=True)
    adopting.add_argument("--dir", type=Path, required=True)
    adopting.add_argument(
        "--bar",
        nargs=5,
        action="append",
        required=True,
        metavar=("ARM", "ENCODER", "CAMPAIGN", "CANDIDATE", "EVENTS"),
    )
    adopting.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    reading = commands.add_parser("read")
    reading.add_argument("--dir", type=Path, required=True)
    reading.add_argument(
        "--partial", action="store_true", help="read a grid not yet whole, as a trial does"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure the pilot's cells, or read the tables from what was measured."""
    arguments = parse_arguments(argv)
    if arguments.command == "measure":  # pragma: no cover - environment
        measure(arguments)
    elif arguments.command == "adopt":
        bars = [
            Bar(
                arm=arm, encoder=encoder, campaign=campaign, candidate=candidate, events=int(events)
            )
            for arm, encoder, campaign, candidate, events in arguments.bar
        ]
        print(f"{adopt(arguments.predictions, arguments.dir, bars, arguments.seeds)} cells adopted")
    else:
        print(read(arguments.dir, partial=arguments.partial))


if __name__ == "__main__":  # pragma: no cover
    main()
