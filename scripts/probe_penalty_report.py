"""Which penalty the closed-form probe chooses on a binary task, under each backbone and pooling.

Over outcomes the probe solved in closed form fits an L2-penalised logistic regression over the
frozen backbone's pooled states, its penalty chosen by the log-loss of five folds within the drawn
labels, among the strengths its settings list. A choice at either end of the list means the list
stops too soon or the labels hold nothing a weaker penalty can use; the folds' loss at every
strength, set beside that of answering with the prevalence, tells the two apart.

This draws the task's budgets from its tuning side as a comparison's cell does, builds the
candidate and embeds the drawn stays by the campaign's own code, and fits the head twice: once
over the whole list, as the candidate does, and once per strength alone on the same folds, so a
strength whose fit does not converge is named rather than taking the draw down with it. No window
of the validation side is read, and nothing is scored.

The numbers are written to CSV first and rendered from that file.

    uv run --env-file .env.r2 scripts/probe_penalty_report.py --manifest KEY CHECKSUM \
        --backbone NAME KEY CHECKSUM [--backbone ...] --out DIR
    uv run scripts/probe_penalty_report.py --report-only DIR
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path
from statistics import fmean
from typing import Self

import torch
from torch import Tensor

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.entrypoints.restored_backbones import RestoredBackbones
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.adapters.torch.logistic_solution import LogisticSolution
from emblema.evaluation.adapters.torch.target_link import TargetLink
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.domain.exceptions import UnsolvableHeadError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.pretraining.adapters.training.devices import available_device
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from scripts.reporting import dated_heading, table

CURVES = "probe_penalties.csv"
TASK = "physionet2012-in-hospital-death"
BUDGETS = ("50", "200", "1000", "all")
SEEDS = (1, 2, 3)
# The strengths the probe chooses among today, a decade apart, one decade below them and three
# above.
IN_FORCE = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
FURTHER = (0.0001, *IN_FORCE, 1e6, 1e7, 1e8)
# The two poolings a selection of the probe on this task would choose between: the mean over the
# window every network started with, and the tail of a fifth the turbofan curve ran under.
POOLINGS = {
    "mean": HeadPooling.mean(),
    "tail-0.2": HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2),
}
# What the worker's schedule holds by default; the closed form reads only the batch, which sets
# how many stays the encoder embeds at once.
SCHEDULE = AdaptationSchedule(
    epochs=30,
    min_steps=2000,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=0.0,
    warmup_fraction=0.1,
    final_lr_fraction=0.01,
)


@dataclass(frozen=True, kw_only=True)
class PenaltyPoint:
    """One strength's fit on one draw, under one backbone and pooling.

    Attributes:
        backbone: The name the backbone was given on the command line.
        pooling: The pooling's name, as ``POOLINGS`` keys it.
        budget: The budget of labels, as the task names it.
        seed: The seed of the draw and of the candidate.
        windows: How many labelled stays the fit learnt from.
        positives: How many of them died.
        penalty: The strength of the L2 penalty.
        log_loss: The folds' mean held-out log-loss at this strength; ``None`` where a fit on a
            fold, or the fit on every stay the candidate would keep, did not converge.
        prevalence_log_loss: The same mean for answering each held-out stay with the prevalence
            of its fold's kept part.
        chosen: The strength the candidate's own fit over the whole list chose; ``None`` where
            that fit did not converge.
        failure: Why this strength's fit failed, or empty.
        embed_seconds: Wall time of embedding the draw's stays.
        fit_seconds: Wall time of the candidate's own fit over the whole list.
    """

    backbone: str
    pooling: str
    budget: str
    seed: int
    windows: int
    positives: int
    penalty: float
    log_loss: float | None
    prevalence_log_loss: float
    chosen: float | None
    failure: str
    embed_seconds: float
    fit_seconds: float

    COLUMNS = (
        "backbone",
        "pooling",
        "budget",
        "seed",
        "windows",
        "positives",
        "penalty",
        "log_loss",
        "prevalence_log_loss",
        "chosen",
        "failure",
        "embed_seconds",
        "fit_seconds",
    )

    @property
    def strongest_in_force(self) -> float:
        return max(IN_FORCE)

    def record(self) -> dict[str, str]:
        values = {column: getattr(self, column) for column in self.COLUMNS}
        return {column: "" if value is None else str(value) for column, value in values.items()}

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            backbone=record["backbone"],
            pooling=record["pooling"],
            budget=record["budget"],
            seed=int(record["seed"]),
            windows=int(record["windows"]),
            positives=int(record["positives"]),
            penalty=float(record["penalty"]),
            log_loss=float(record["log_loss"]) if record["log_loss"] else None,
            prevalence_log_loss=float(record["prevalence_log_loss"]),
            chosen=float(record["chosen"]) if record["chosen"] else None,
            failure=record["failure"],
            embed_seconds=float(record["embed_seconds"]),
            fit_seconds=float(record["fit_seconds"]),
        )


def prevalence_log_loss(outcomes: Sequence[float]) -> float:
    """The folds' mean held-out log-loss of answering with the prevalence of the kept part."""
    folds = OutcomeFolds.of(list(outcomes))
    losses = []
    for fold in range(folds.count):
        kept = [outcomes[index] for index in folds.kept(fold)]
        held = [outcomes[index] for index in folds.held_out(fold)]
        share = fmean(kept)
        losses.append(fmean(-log(share) if died else -log(1.0 - share) for died in held))
    return fmean(losses)


class Sampler:
    """The task's draws from its tuning side, embedded by a frozen backbone as a cell embeds them.

    Built once over the published corpus, so that every draw of a report reads the same task,
    labels and blocks; a backbone is loaded once and serves every draw and pooling asked of it.
    """

    def __init__(
        self,
        *,
        store_settings: Settings,
        manifest: ArtifactRef,
        corpora: Path,
        workspace: Path,
        device: str,
    ) -> None:
        self._store = configured_store(store_settings)
        known = KnownTasks.named(TASK)
        self._blocks = PublishedCorpusBlocks(self._store, workspace)
        corpus = BlockCorpusWindows(self._blocks)
        tasks = InMemoryDownstreamTaskRepository()
        self._task_id = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
            known.defined_over(manifest, corpus.describe(manifest))
        )
        task = tasks.get(self._task_id)
        self._draw = DrawLabelBudget(tasks, corpus, KnownGroundTruths.under(corpora))
        self._manifest = self._blocks.manifest_of(task.manifest)
        self._block = self._blocks.block_of(self._manifest)
        self._link = TargetLink.of(task.label_scheme())
        self._device = device
        self._backbones: dict[ArtifactRef, RestoredBackbones] = {}

    def drawn(self, budget: str, seed: int) -> tuple[list[TokenWindow], list[float], float]:
        """The stays of one draw, their outcomes as the head learns them, and their mean label."""
        sample = self._draw(
            DrawLabelBudgetCommand(task=self._task_id, budget=LabelBudget.parse(budget), seed=seed)
        )
        windows = list(self._block.at([labelled.window.position for labelled in sample.windows]))
        outcomes = [self._link.learnt(labelled.target) for labelled in sample.windows]
        return windows, outcomes, sample.mean_target

    def embedded(
        self,
        weights: ArtifactRef,
        pooling: HeadPooling,
        seed: int,
        windows: Sequence[TokenWindow],
        mean_target: float,
    ) -> Tensor:
        """The pooled states the closed-form probe reads, built and embedded as a cell does."""
        if weights not in self._backbones:
            self._backbones[weights] = RestoredBackbones(self._store, weights)
        plan = AdaptationPlan(
            mode=TransferMode.FROZEN_RIDGE,
            backbone=weights,
            schedule=SCHEDULE,
            lora=None,
            seed=seed,
            pooling=pooling,
            ridge=RidgePenalties(IN_FORCE),
        )
        torch.manual_seed(seed)
        candidate = AdaptedBackbone.under(
            plan,
            self._backbones[weights],
            vocabulary_size=len(self._manifest.channels),
            starting_at=self._link.starting_at(mean_target),
        ).to(self._device)
        candidate.eval()
        size = SCHEDULE.batch_size
        with torch.no_grad():
            states = [
                candidate.embed(
                    TokenTensors.from_windows(list(windows[start : start + size])).to(self._device)
                )
                for start in range(0, len(windows), size)
            ]
        return torch.cat(states)


def trace(
    sampler: Sampler,
    backbones: Sequence[tuple[str, ArtifactRef]],
    budgets: Sequence[str],
    seeds: Sequence[int],
    penalties: Sequence[float],
) -> list[PenaltyPoint]:
    """Every strength's folds' log-loss on every draw, backbone and pooling, beside the choice."""
    points = []
    for budget in budgets:
        for seed in seeds:
            windows, outcomes, mean_target = sampler.drawn(budget, seed)
            taught = torch.tensor(outcomes, dtype=torch.float64)
            baseline = prevalence_log_loss(outcomes)
            for name, weights in backbones:
                for pooled, pooling in POOLINGS.items():
                    started = time.perf_counter()
                    states = sampler.embedded(weights, pooling, seed, windows, mean_target)
                    embed_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    chosen: float | None
                    try:
                        chosen = LogisticSolution.fitted(
                            states, taught, RidgePenalties(tuple(penalties))
                        ).penalty
                    except UnsolvableHeadError:
                        chosen = None
                    fit_seconds = time.perf_counter() - started
                    for penalty in penalties:
                        try:
                            solved = LogisticSolution.fitted(
                                states, taught, RidgePenalties((penalty,))
                            )
                            loss, failure = solved.held_out_loss[0], ""
                        except UnsolvableHeadError as error:
                            loss, failure = None, str(error)
                        points.append(
                            PenaltyPoint(
                                backbone=name,
                                pooling=pooled,
                                budget=budget,
                                seed=seed,
                                windows=len(outcomes),
                                positives=int(sum(outcomes)),
                                penalty=penalty,
                                log_loss=loss,
                                prevalence_log_loss=baseline,
                                chosen=chosen,
                                failure=failure,
                                embed_seconds=embed_seconds,
                                fit_seconds=fit_seconds,
                            )
                        )
                    print(
                        f"{name} {pooled} {budget} seed {seed}: {len(outcomes)} stays, penalty "
                        f"{'failed' if chosen is None else f'{chosen:g}'}, embedded in "
                        f"{embed_seconds:.0f} s, fitted in {fit_seconds:.1f} s",
                        flush=True,
                    )
    return points


def write(rows: Sequence[PenaltyPoint], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CURVES).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PenaltyPoint.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.record() for row in rows)
    return directory / CURVES


def read(directory: Path) -> tuple[PenaltyPoint, ...]:
    """The points stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CURVES).is_file():
        raise SystemExit(f"{directory} holds no {CURVES}; nothing to render")
    with (directory / CURVES).open(newline="", encoding="utf-8") as handle:
        return tuple(PenaltyPoint.parse(row) for row in csv.DictReader(handle))


def render(points: Sequence[PenaltyPoint]) -> str:
    """One row per draw, backbone and pooling: the choice, the folds' best, the prevalence's."""
    keys = list(dict.fromkeys((p.backbone, p.pooling, p.budget, p.seed) for p in points))
    rows = []
    for key in keys:
        drawn = [p for p in points if (p.backbone, p.pooling, p.budget, p.seed) == key]
        solved = [p for p in drawn if p.log_loss is not None]
        best = min(solved, key=lambda p: (p.log_loss, p.penalty)) if solved else None
        first = drawn[0]
        edge = (
            "failed"
            if first.chosen is None
            else "strongest"
            if first.chosen >= first.strongest_in_force
            else "weakest"
            if first.chosen <= min(IN_FORCE)
            else "inside"
        )
        rows.append(
            (
                first.backbone,
                first.pooling,
                first.budget,
                str(first.seed),
                str(first.positives),
                "failed" if first.chosen is None else f"{first.chosen:g}",
                "—" if best is None else f"{best.penalty:g} ({best.log_loss:.4f})",
                f"{first.prevalence_log_loss:.4f}",
                edge,
                ", ".join(f"{p.penalty:g}" for p in drawn if p.failure) or "—",
                f"{first.embed_seconds:.0f}",
                f"{first.fit_seconds:.1f}",
            )
        )
    return "\n\n".join(
        [
            f"Task {TASK}, tuning side only; strengths in force {min(IN_FORCE):g} to "
            f"{max(IN_FORCE):g}.",
            table(
                (
                    "Backbone",
                    "Pooling",
                    "Budget",
                    "Seed",
                    "Deaths",
                    "Chosen",
                    "Folds' best (log-loss)",
                    "Prevalence alone",
                    "Against the list in force",
                    "Strengths that failed",
                    "Embed s",
                    "Fit s",
                ),
                rows,
            ),
        ]
    )


def report(directory: Path) -> str:
    return "\n\n".join([dated_heading(), render(read(directory))])


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", nargs=2, metavar=("KEY", "CHECKSUM"))
    parser.add_argument(
        "--backbone",
        nargs=3,
        action="append",
        metavar=("NAME", "KEY", "CHECKSUM"),
        help="a backbone's weights and the name the tables give it; repeat for several",
    )
    parser.add_argument("--out", type=Path, help="directory the CSV is written to")
    parser.add_argument(
        "--report-only", type=Path, help="render the table from a directory written earlier"
    )
    parser.add_argument("--budget", action="append", help="a budget; every one by default")
    parser.add_argument("--seed", action="append", type=int, help="a seed; 1 to 3 by default")
    parser.add_argument("--device", default=available_device())
    parser.add_argument("--corpora", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "workspace")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(report(arguments.report_only))
        return
    if arguments.out is None or arguments.manifest is None or not arguments.backbone:
        raise SystemExit(
            "--out DIR, --manifest KEY CHECKSUM and --backbone NAME KEY CHECKSUM are required, "
            "or --report-only DIR"
        )
    key, checksum = arguments.manifest
    sampler = Sampler(
        store_settings=Settings(),
        manifest=ArtifactRef(key, Checksum.parse(checksum)),
        corpora=arguments.corpora,
        workspace=arguments.workspace,
        device=arguments.device,
    )
    backbones = [
        (name, ArtifactRef(weights, Checksum.parse(digest)))
        for name, weights, digest in arguments.backbone
    ]
    points = trace(
        sampler, backbones, arguments.budget or BUDGETS, arguments.seed or SEEDS, FURTHER
    )
    write(points, arguments.out)
    print(report(arguments.out))


if __name__ == "__main__":
    main()
