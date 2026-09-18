r"""Adapt the pretrained backbone to the turbofan task under every transfer mode, one run each.

The first numbers of the label-efficiency comparison, before the grid: each mode learns the task
from one budget of labels drawn under one seed and answers the validation engines. Runs are
stored as CSV — a row per run, a row per epoch, a row per validation window — and the note's
table is rendered from those files, never from what is still in memory, so a change of wording
costs a rerender and not a run.

    uv run --env-file .env.r2 scripts/transfer_modes_report.py \
        --weights <key> <checksum> --manifest <key> <checksum> --budget 200 --device mps

    uv run scripts/transfer_modes_report.py --report-only data/report/transfer/<run>

The task is the turbofan one the preregistration names; which engines make it up, its label
ceiling and its strata are stated in ``KnownTasks`` here until a process of the Evaluation
context owns them. The ground truth is read from the downloaded corpus under ``data/raw``.
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Self

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.configured import configured_store
from emblema.entrypoints.cli.pretrain.source_revision import SourceRevision
from emblema.entrypoints.cli.restored_backbones import RestoredBackbones
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.readers.cmapss_unit_lifetimes import CmapssUnitLifetimes
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import TorchAdaptationRuntime
from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTask,
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.raw_corpora import raw_root
from scripts.reporting import dated_heading, machine, table

RUNS, EPOCHS, PREDICTIONS = "runs.csv", "epochs.csv", "predictions.csv"
RUN_COLUMNS = (
    "mode",
    "backbone",
    "run_seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
    "budget",
    "sample_seed",
    "trainable_parameters",
    "rmse",
    "seconds",
    "device",
    "commit",
    "torch",
)


@dataclass(frozen=True, kw_only=True)
class KnownTask:
    """What a supervised task is made of, as far as no adapter can read it off the corpus.

    Attributes:
        name: What the task is called in the reports.
        corpus: Name the corpus was published under.
        unit_prefix: What the keys of the task's units start with, inside that corpus.
        ceiling: Largest remaining life a window is labelled with.
        strata: How many groups of the target a budget is spread over.
        test_source: Where the frozen test engines come from, as the set is known outside.
        test_units: Keys of the frozen test engines.
    """

    name: str
    corpus: str
    unit_prefix: str
    ceiling: float
    strata: int
    test_source: str
    test_units: tuple[str, ...]

    def units_of(self, keys: Sequence[str]) -> frozenset[UnitKey]:
        """The task's units among the keys a published corpus names."""
        return frozenset(UnitKey(key) for key in keys if key.startswith(self.unit_prefix))


class KnownTasks:
    """The tasks a report may run, each stated once."""

    TURBOFAN_FD001 = KnownTask(
        name="turbofan-fd001",
        corpus="cmapss",
        unit_prefix="FD001/",
        ceiling=125.0,
        strata=4,
        test_source="cmapss/test/FD001",
        test_units=tuple(f"FD001/test/{engine}" for engine in range(1, 101)),
    )

    @classmethod
    def default(cls) -> KnownTask:
        return cls.TURBOFAN_FD001


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored run")
    parser.add_argument("--weights", nargs=2, metavar=("KEY", "CHECKSUM"), help="the backbone")
    parser.add_argument("--manifest", nargs=2, metavar=("KEY", "CHECKSUM"), help="the corpus")
    parser.add_argument(
        "--mode",
        action="append",
        type=TransferMode,
        choices=list(TransferMode),
        help="a mode to run; every mode unless given",
    )
    parser.add_argument("--budget", default="200", help="labelled windows, or 'all'")
    parser.add_argument(
        "--sample-seed", type=int, default=1, help="seed the labels are drawn under"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed of the run: the head, the fresh weights, the updates, the order of windows",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    for mode, rate in (
        (TransferMode.FROM_SCRATCH, 1e-3),
        (TransferMode.FROZEN_PROBE, 1e-2),
        (TransferMode.LORA, 1e-3),
        (TransferMode.FULL_FINE_TUNING, 1e-4),
    ):
        parser.add_argument(
            f"--lr-{mode.value.replace('_', '-')}",
            type=float,
            default=rate,
            help=f"learning rate of {mode.value}",
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
        help="where the run is stored; data/report/transfer/<timestamp> unless given",
    )
    return parser.parse_args(argv)


def ref_of(pair: Sequence[str]) -> ArtifactRef:
    key, checksum = pair
    return ArtifactRef(key, Checksum.parse(checksum))


def budget_of(text: str) -> LabelBudget:
    return LabelBudget.everything() if text == "all" else LabelBudget.of(int(text))


def plans_of(arguments: argparse.Namespace) -> dict[TransferMode, AdaptationPlan]:
    """One plan per mode asked for, every knob read off the arguments."""
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
                batch_size=arguments.batch_size,
                learning_rate=getattr(arguments, f"lr_{mode.value}"),
                weight_decay=arguments.weight_decay,
            ),
            lora=lora if mode.adds_low_rank_updates else None,
            seed=arguments.seed,
        )
        for mode in modes
    }


@dataclass(frozen=True, kw_only=True)
class Stored:
    """Where one report's runs are kept, and how a run is appended to them.

    One directory holds one report: the epochs and the predictions are keyed by mode alone, so a
    second report appended into the same files could not be told from the first.
    """

    directory: Path

    @classmethod
    def at(cls, directory: Path) -> Self:
        """A directory for a report about to run.

        Raises:
            SystemExit: If a report is already stored there.
        """
        if (directory / RUNS).exists():
            raise SystemExit(
                f"{directory} already holds {RUNS}; choose another --out, "
                "or --report-only to render it"
            )
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory=directory)

    @classmethod
    def existing(cls, directory: Path) -> Self:
        """A report already stored, to be rendered again.

        Raises:
            SystemExit: If nothing is stored there.
        """
        if not (directory / RUNS).is_file():
            raise SystemExit(f"{directory} holds no {RUNS}; nothing to render")
        return cls(directory=directory)

    def add(self, outcome: AdaptationOutcome, *, device: str, commit: str) -> None:
        """Append the run, its epochs and its predictions to the three files."""
        row = {
            **outcome.plan.parameters(),
            "budget": "all" if outcome.budget.windows is None else outcome.budget.windows,
            "sample_seed": outcome.sample_seed,
            "trainable_parameters": outcome.trainable_parameters,
            "rmse": f"{outcome.rmse:.6f}",
            "seconds": f"{outcome.seconds:.3f}",
            "device": device,
            "commit": commit,
            "torch": version("torch"),
        }
        self._append(RUNS, RUN_COLUMNS, [row])
        mode = str(outcome.plan.mode)
        self._append(
            EPOCHS,
            ("mode", "epoch", "loss"),
            [
                {"mode": mode, "epoch": epoch, "loss": f"{loss:.8f}"}
                for epoch, loss in enumerate(outcome.training_losses)
            ],
        )
        self._append(
            PREDICTIONS,
            ("mode", "unit", "position", "ends_at", "target", "predicted"),
            [
                {
                    "mode": mode,
                    "unit": str(p.window.unit),
                    "position": p.window.position,
                    "ends_at": p.window.ends_at,
                    "target": p.target,
                    "predicted": f"{p.predicted:.6f}",
                }
                for p in outcome.predictions
            ],
        )

    def runs(self) -> list[dict[str, str]]:
        with (self.directory / RUNS).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _append(
        self, name: str, columns: Sequence[str], rows: Sequence[Mapping[str, object]]
    ) -> None:
        path = self.directory / name
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerows(rows)


def render(stored: Stored, task: KnownTask) -> str:
    """The section a note pastes: one row per run, read back from what was stored."""
    runs = stored.runs()
    lines = [
        dated_heading(),
        "",
        f"Task `{task.name}`, ceiling {task.ceiling:g}, {task.strata} strata; "
        f"machine: {machine()}.",
        "Every number is validation, not test.",
        "",
        table(
            (
                "mode",
                "budget",
                "run seed",
                "epochs",
                "lr",
                "trainable",
                "RMSE",
                "seconds",
                "device",
            ),
            [
                (
                    run["mode"],
                    run["budget"],
                    run["run_seed"],
                    run["epochs"],
                    run["learning_rate"],
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
    task = KnownTasks.default()
    if arguments.report_only is not None:
        print(render(Stored.existing(arguments.report_only), task))
        return
    if arguments.weights is None or arguments.manifest is None:
        raise SystemExit("--weights and --manifest are required to run; --report-only to render")
    root = raw_root(task.corpus)
    if root is None:
        raise SystemExit(f"the {task.corpus} corpus is not under data/raw; fetch it first")
    device = available_device() if arguments.device is None else arguments.device
    out = arguments.out or Path("data/report/transfer") / datetime.now().strftime("%Y%m%d-%H%M%S")
    stored = Stored.at(out)
    commit = SourceRevision().current()

    store = configured_store(Settings())
    manifest = ref_of(arguments.manifest)
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    corpus = BlockCorpusWindows(blocks)
    lifetimes = CmapssUnitLifetimes(root)
    tasks = InMemoryDownstreamTaskRepository()
    sides = corpus.describe(manifest)
    task_id = DefineDownstreamTask(tasks, corpus, Uuid4IdGenerator())(
        DefineDownstreamTaskCommand(
            manifest=manifest,
            units=task.units_of(sorted(str(u) for u in sides.training | sides.validation)),
            test=FrozenTestSplit(
                units=frozenset(UnitKey(key) for key in task.test_units), source=task.test_source
            ),
            labels=RemainingLifeScheme(task.ceiling),
            strata=TargetBins(task.strata),
        )
    )
    runtime = TorchAdaptationRuntime(
        RestoredBackbones(store, ref_of(arguments.weights)), blocks, device=device
    )
    run = RunAdaptation(
        tasks, corpus, lifetimes, DrawLabelBudget(tasks, corpus, lifetimes), runtime
    )
    budget = budget_of(arguments.budget)
    for mode, plan in plans_of(arguments).items():
        print(f"{mode}: learning on {device} ...", flush=True)
        outcome = run(
            RunAdaptationCommand(
                task=task_id, plan=plan, budget=budget, sample_seed=arguments.sample_seed
            )
        )
        stored.add(outcome, device=device, commit=commit)
        print(
            f"{mode}: RMSE {outcome.rmse:.2f} over {len(outcome.predictions)} windows, "
            f"{outcome.trainable_parameters} trainable, {outcome.seconds:.0f} s",
            flush=True,
        )
    print()
    print(render(stored, task))
    print(f"\nstored under {out}")


if __name__ == "__main__":
    main()
