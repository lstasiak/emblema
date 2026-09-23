r"""Adapt the pretrained backbone to the turbofan task, one cell of the curve at a time.

A cell is a transfer mode at a budget of labels under a seed; the grid the curve is drawn from
is every mode at every budget under every seed. Cells are stored as they finish — a row per run,
a row per epoch, a row per validation window — so a session that drops leaves what it did, and
the next one skips the cells already on file, refusing to go on under another configuration.
Tables are rendered from those files, never from what is still in memory, so a change of
wording costs a rerender and not a run.

One cell, or a few, on this machine (the budget of 200 is the tier the machine serves):

    uv run --env-file .env.r2 scripts/transfer_modes_report.py \
        --weights <key> <checksum> --manifest <key> <checksum> \
        --budget 200 --seed 1 --mode from_scratch --device mps

The grid on a platform with two accelerators, sharded by seed and published to the bucket
after every cell, so a session that drops has lost one cell at most:

    python scripts/transfer_modes_report.py --weights ... --manifest ... \
        --seed 1 --seed 3 --seed 5 --device cuda:0 --out shard-0 --publish

A session started afresh resumes from the last reference the log printed: fetched into the
same directory, the same command skips the cells it holds. Back here, fetched and rendered:

    uv run --env-file .env.r2 scripts/transfer_modes_report.py --fetch <key> <checksum> --out DIR
    uv run scripts/transfer_modes_report.py --report-only DIR

The task is the turbofan one the preregistration names; which engines make it up, its label
ceiling and its strata are stated in ``KnownTasks`` (``scripts/transfer_grid.py``) until a
process of the Evaluation context owns them. The ground truth is read from the downloaded
corpus under ``data/raw``. One seed drives both the draw of the labels and the run, so the
repeats of a cell differ in which labels they held as well as in what they learnt from them.
"""

import argparse
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.known_tasks import KnownTask, KnownTasks
from emblema.entrypoints.cli.pretrain.source_revision import SourceRevision
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.adapters.synthetic.synthetic_ground_truth import SyntheticGroundTruth
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import TorchAdaptationRuntime
from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTask,
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.ground_truth import GroundTruth
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.shared.adapters.synthetic.sensor_signal import SensorSignal
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.raw_corpora import raw_root
from scripts.reporting import dated_heading, machine, table
from scripts.transfer_grid import BUDGETS, SEEDS, Cell, Stored, budget_of

# The peak rate of each mode, as fixed on the validation side before the grid
# (docs/verification/label-efficiency-curve.md); the shape of the rate is fixed with them.
LEARNING_RATES = {
    TransferMode.FROM_SCRATCH: 3e-4,
    TransferMode.FROZEN_PROBE: 1e-2,
    TransferMode.LORA: 3e-3,
    TransferMode.FULL_FINE_TUNING: 3e-4,
}
WARMUP_FRACTION, FINAL_LR_FRACTION = 0.1, 0.01


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored run")
    parser.add_argument(
        "--fetch", nargs=2, metavar=("KEY", "CHECKSUM"), help="unpack a published run into --out"
    )
    parser.add_argument(
        "--task",
        choices=KnownTasks.names(),
        default=KnownTasks.default().name,
        help="the supervised task the grid is run on",
    )
    parser.add_argument("--weights", nargs=2, metavar=("KEY", "CHECKSUM"), help="the backbone")
    parser.add_argument("--manifest", nargs=2, metavar=("KEY", "CHECKSUM"), help="the corpus")
    parser.add_argument(
        "--mode",
        action="append",
        type=TransferMode,
        choices=list(TransferMode),
        help="a mode to run; every mode unless given",
    )
    parser.add_argument(
        "--budget",
        action="append",
        help=f"labelled windows, or 'all'; may repeat; {' '.join(BUDGETS)} unless given",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help=(
            "seed of a cell: the draw of the labels, the head, the fresh weights, the updates, "
            f"the order of windows; may repeat; {' '.join(map(str, SEEDS))} unless given"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--min-steps",
        type=int,
        default=2000,
        help="optimiser steps a run takes at least, in whole epochs; 0 for the epochs alone",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=WARMUP_FRACTION,
        help="share of the run's steps the rate climbs over; the same for every mode",
    )
    parser.add_argument(
        "--final-lr-fraction",
        type=float,
        default=FINAL_LR_FRACTION,
        help="fraction of the peak the rate decays to; one for a constant rate",
    )
    for mode, rate in LEARNING_RATES.items():
        parser.add_argument(
            f"--lr-{mode.value.replace('_', '-')}",
            type=float,
            default=rate,
            help=f"peak learning rate of {mode.value}",
        )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=["qkv", "attention.projection", "feedforward"],
        help="runs of the layer paths the low-rank updates go beside",
    )
    parser.add_argument("--device", default=None, help="where to compute; the accelerator if any")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/artifacts"),
        help="where blocks are fetched to and mapped from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the run is stored or unpacked; data/report/transfer/<timestamp> unless given",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="after every cell, store the directory in the bucket and print its reference",
    )
    return parser.parse_args(argv)


def ground_truth_of(task: KnownTask) -> GroundTruth:
    """Where the task's labels are read from.

    The raw files of a real corpus, the generator of a synthetic one.

    Raises:
        SystemExit: If the task's ground truth is not at hand on this machine.
    """
    match task.labels:
        case RemainingLifeScheme():
            if task.corpus != "cmapss":
                raise SystemExit(f"no reader of failures is known for the {task.corpus} corpus")
            root = raw_root(task.corpus)
            if root is None:
                raise SystemExit(f"the {task.corpus} corpus is not under data/raw; fetch it first")
            return CmapssGroundTruth(root)
        case ForecastScheme():
            if task.corpus not in LAYOUTS:
                raise SystemExit(f"no generated layout is called {task.corpus!r}")
            return SyntheticGroundTruth(
                SensorSignal(CONTROL_PROCESS, LAYOUTS[task.corpus]), task.labels
            )


def ref_of(pair: Sequence[str]) -> ArtifactRef:
    key, checksum = pair
    return ArtifactRef(key, Checksum.parse(checksum))


def plans_of(arguments: argparse.Namespace, seed: int) -> dict[TransferMode, AdaptationPlan]:
    """One plan per mode asked for under ``seed``, every knob read off the arguments."""
    weights = ref_of(arguments.weights)
    lora = LoraSpec(
        rank=arguments.lora_rank,
        alpha=arguments.lora_alpha,
        dropout=arguments.lora_dropout,
        targets=tuple(arguments.lora_targets),
    )
    modes = arguments.mode or list(TransferMode)
    return {
        mode: AdaptationPlan(
            mode=mode,
            backbone=weights if mode.starts_from_pretrained_weights else None,
            schedule=AdaptationSchedule(
                epochs=arguments.epochs,
                min_steps=arguments.min_steps,
                batch_size=arguments.batch_size,
                learning_rate=getattr(arguments, f"lr_{mode.value}"),
                weight_decay=arguments.weight_decay,
                warmup_fraction=arguments.warmup_fraction,
                final_lr_fraction=arguments.final_lr_fraction,
            ),
            lora=lora if mode.adds_low_rank_updates else None,
            seed=seed,
        )
        for mode in modes
    }


def cells_of(arguments: argparse.Namespace) -> Iterator[tuple[Cell, AdaptationPlan]]:
    """Every cell asked for, the cheapest budgets first so a session reports early."""
    budgets = [budget_of(text) for text in (arguments.budget or BUDGETS)]
    for seed in arguments.seed or SEEDS:
        plans = plans_of(arguments, seed)
        for budget in budgets:
            for mode, plan in plans.items():
                yield Cell(mode=mode, budget=budget, seed=seed), plan


def render(stored: Stored, task: KnownTask) -> str:
    """The section a note pastes: one row per cell, read back from what was stored."""
    runs = stored.runs()
    lines = [
        dated_heading(),
        "",
        f"Task `{task.name}`, {task.labels_text}, {task.strata} strata; machine: {machine()}.",
        "Every number is validation, not test.",
        "",
        table(
            (
                "mode",
                "budget",
                "windows",
                task.units_called,
                "seed",
                "epochs",
                "steps",
                "lr",
                "warm-up",
                "final lr",
                "trainable",
                "RMSE",
                "seconds",
                "device",
            ),
            [
                (
                    run["mode"],
                    run["budget"],
                    # Runs stored before a column existed render it empty rather than not at all.
                    run.get("windows", ""),
                    run.get("engines", ""),
                    run["sample_seed"],
                    run["epochs"],
                    run.get("steps", ""),
                    run["learning_rate"],
                    run.get("warmup_fraction", ""),
                    run.get("final_lr_fraction", ""),
                    run["trainable_parameters"],
                    f"{float(run['rmse']):.2f}",
                    f"{float(run['seconds']):.0f}",
                    run["device"],
                )
                for run in runs
            ],
        ),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse(argv)
    task = KnownTasks.named(arguments.task)
    if arguments.report_only is not None:
        print(render(Stored.existing(arguments.report_only), task))
        return
    if arguments.fetch is not None:
        if arguments.out is None:
            raise SystemExit("--fetch needs --out, the directory to unpack into")
        stored = Stored.fetch(configured_store(Settings()), ref_of(arguments.fetch), arguments.out)
        print(render(stored, task))
        return
    if arguments.weights is None or arguments.manifest is None:
        raise SystemExit("--weights and --manifest are required to run; --report-only to render")
    truth = ground_truth_of(task)
    device = available_device() if arguments.device is None else arguments.device
    out = arguments.out or Path("data/report/transfer") / datetime.now().strftime("%Y%m%d-%H%M%S")
    stored = Stored.open(out)
    commit = SourceRevision().current()
    stored.require_configuration(task=task.name, commit=commit)

    store = configured_store(Settings())
    manifest = ref_of(arguments.manifest)
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    sides = corpus.describe(manifest)
    test = task.frozen_test(task.units_of(sorted(str(u) for u in sides.validation)))
    task_id = DefineDownstreamTask(tasks, corpus, Uuid4IdGenerator())(
        DefineDownstreamTaskCommand(
            manifest=manifest,
            units=task.units_of(sorted(str(u) for u in sides.training | sides.validation))
            - test.units,
            test=test,
            protocol=EvaluationProtocol.LABEL_BUDGET,
            labels=task.labels,
            strata=TargetBins(task.strata),
        )
    )
    runtime = TorchAdaptationRuntime(
        RestoredBackbones(store, ref_of(arguments.weights)), blocks, device=device
    )
    ids = Uuid4IdGenerator()
    # A tuning report never opens the frozen side, but the use case is what decides that, so it
    # is given the operation rather than trusted not to need it.
    open_test = OpenTestSplit(
        tasks, ids, SystemClock(), InMemoryEventPublisher(InMemoryEventSubscriber())
    )
    run = RunAdaptation(
        DrawRunLabels(tasks, corpus, truth, DrawLabelBudget(tasks, corpus, truth), open_test),
        runtime,
    )
    try:
        for cell, plan in cells_of(arguments):
            if stored.holds(cell, plan, task=task.name, commit=commit):
                print(f"{cell}: already stored, skipped", flush=True)
                continue
            print(f"{cell}: learning on {device} ...", flush=True)
            outcome = run(
                RunAdaptationCommand(
                    task=task_id, plan=plan, budget=cell.budget, sample_seed=cell.seed
                )
            )
            stored.add(outcome, task=task.name, device=device, commit=commit)
            print(
                f"{cell}: RMSE {outcome.rmse:.2f} over {len(outcome.predictions)} windows, "
                f"{outcome.labelled_units} {task.units_called} labelled, "
                f"{outcome.trainable_parameters} trainable, {outcome.seconds:.0f} s",
                flush=True,
            )
            if arguments.publish:
                published = stored.publish(store)
                print(f"{cell}: published as {published.key} {published.checksum}", flush=True)
    finally:
        # A cell that raised is not stored; what is stored is rendered and published whatever
        # stopped the loop, so a session that ends early still leaves its reference in the log.
        print()
        print(render(stored, task))
        print(f"\nstored under {out}")
        if arguments.publish and stored.runs():
            published = stored.publish(store)
            print(f"published as\n{published.key}\n{published.checksum}")


if __name__ == "__main__":
    main()
