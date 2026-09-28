"""Which penalty MiniRocket's logistic regression chooses on a binary task, and what it costs.

Over outcomes the convolution baseline fits an L2-penalised logistic regression, its penalty
chosen by the log-loss of folds within the drawn labels, among the strengths its settings list.
A choice at the strongest strength listed says the list stops too soon. This draws the task's
budgets from its tuning side exactly as a campaign's cell does, fits the candidate by the same
code over a list of strengths that runs further, and records which one each fit chose and how
long the fit took. No window of the validation side is read: the choice is made within the
drawn labels, and nothing is scored.

The numbers are written to CSV first and rendered from that file.

    uv run scripts/convolution_penalty_report.py --manifest KEY CHECKSUM --out DIR
    uv run scripts/convolution_penalty_report.py --report-only DIR
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.minirocket.fitted_convolutions import FittedConvolutions
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.classical.ridge_spec import RidgeSpec
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.reporting import dated_heading, table

CHOICES = "penalty_choices.csv"
TASK = "physionet2012-in-hospital-death"
BUDGETS = ("50", "200", "1000", "all")
SEEDS = (1, 2, 3)
# The strengths the candidate chooses among today, a factor of about 4.64 apart, and four more
# steps of the same factor above the strongest.
IN_FORCE = (0.001, 0.00464, 0.0215, 0.1, 0.464, 2.15, 10.0, 46.4, 215.0, 1000.0)
FURTHER = (*IN_FORCE, 4640.0, 21500.0, 100000.0, 464000.0)
FEATURES = 9996


@dataclass(frozen=True, kw_only=True)
class PenaltyChoice:
    """What one fit chose.

    Attributes:
        budget: The budget of labels, as the task names it.
        seed: The seed of the draw and of the convolutions.
        windows: How many labelled windows the fit learnt from.
        positives: How many of them hold the positive outcome.
        chosen: The penalty the folds' log-loss chose among those offered.
        strongest_in_force: The strongest penalty the candidate is offered today.
        seconds: Wall time of the fit, convolutions included.
    """

    budget: str
    seed: int
    windows: int
    positives: int
    chosen: float
    strongest_in_force: float
    seconds: float

    COLUMNS = (
        "budget",
        "seed",
        "windows",
        "positives",
        "chosen",
        "strongest_in_force",
        "seconds",
    )

    @property
    def beyond_the_grid_in_force(self) -> bool:
        """Whether the choice is stronger than anything offered today, or at its edge."""
        return self.chosen >= self.strongest_in_force

    def record(self) -> dict[str, str]:
        return {column: str(getattr(self, column)) for column in self.COLUMNS}

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            budget=record["budget"],
            seed=int(record["seed"]),
            windows=int(record["windows"]),
            positives=int(record["positives"]),
            chosen=float(record["chosen"]),
            strongest_in_force=float(record["strongest_in_force"]),
            seconds=float(record["seconds"]),
        )


def choose(
    *,
    store_settings: Settings,
    manifest: ArtifactRef,
    corpora: Path,
    workspace: Path,
    budgets: Sequence[str],
    seeds: Sequence[int],
    penalties: Sequence[float],
    features: int,
    threads: int,
) -> list[PenaltyChoice]:
    """Draw every budget under every seed from the tuning side, fit, and record the choice."""
    known = KnownTasks.named(TASK)
    blocks = PublishedCorpusBlocks(configured_store(store_settings), workspace)
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    task_id = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        known.defined_over(manifest, corpus.describe(manifest))
    )
    task = tasks.get(task_id)
    draw = DrawLabelBudget(tasks, corpus, KnownGroundTruths.under(corpora))
    read = ReadCorpus.every(blocks, (task,))[task.manifest]
    scheme = task.label_scheme()
    method = RandomConvolutions(
        convolutions=MiniRocketSpec(features=features),
        ridge=RidgeSpec(penalties=tuple(penalties), threads=threads),
    )
    steps = method.convolutions.steps_over(read.manifest.window_length)
    choices = []
    for budget in budgets:
        for seed in seeds:
            sample = draw(
                DrawLabelBudgetCommand(task=task_id, budget=LabelBudget.parse(budget), seed=seed)
            )
            targets = np.array([labelled.target / scheme.scale for labelled in sample.windows])
            started = time.perf_counter()
            fitted = FittedConvolutions.fitted(
                ClassicalRecipe(method=method, seed=seed, sources=()),
                method,
                read.channels,
                steps,
                list(read.windows([labelled.window for labelled in sample.windows])),
                targets,
                scheme.scale,
                scheme.kind,
            )
            choice = PenaltyChoice(
                budget=budget,
                seed=seed,
                windows=len(sample.windows),
                positives=int(targets.sum()),
                chosen=fitted.penalty,
                strongest_in_force=max(IN_FORCE),
                seconds=time.perf_counter() - started,
            )
            print(
                f"{budget} seed {seed}: {choice.windows} windows, penalty {choice.chosen:g}, "
                f"{choice.seconds:.0f} s",
                flush=True,
            )
            choices.append(choice)
    return choices


def write(rows: Sequence[PenaltyChoice], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CHOICES).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PenaltyChoice.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.record() for row in rows)
    return directory / CHOICES


def read_choices(directory: Path) -> tuple[PenaltyChoice, ...]:
    """The choices stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CHOICES).is_file():
        raise SystemExit(f"{directory} holds no {CHOICES}; nothing to render")
    with (directory / CHOICES).open(newline="", encoding="utf-8") as handle:
        return tuple(PenaltyChoice.parse(row) for row in csv.DictReader(handle))


def render(rows: Sequence[PenaltyChoice]) -> str:
    """The note's table: one row per fit, the grid in force marked where a choice passes it."""
    return "\n\n".join(
        [
            dated_heading(),
            f"Task {TASK}, tuning side only; strongest penalty in force "
            f"{rows[0].strongest_in_force:g}.",
            table(
                (
                    "Budget",
                    "Seed",
                    "Windows",
                    "Positives",
                    "Penalty chosen",
                    "At or past the edge",
                    "Seconds",
                ),
                (
                    (
                        row.budget,
                        str(row.seed),
                        f"{row.windows:,}",
                        str(row.positives),
                        f"{row.chosen:g}",
                        "yes" if row.beyond_the_grid_in_force else "no",
                        f"{row.seconds:.0f}",
                    )
                    for row in rows
                ),
            ),
        ]
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", nargs=2, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--out", type=Path, help="directory the CSV is written to")
    parser.add_argument(
        "--report-only", type=Path, help="render the table from a directory written earlier"
    )
    parser.add_argument("--budget", action="append", help="a budget; every one by default")
    parser.add_argument("--seed", action="append", type=int, help="a seed; 1 to 3 by default")
    parser.add_argument("--features", type=int, default=FEATURES)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--corpora", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "workspace")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(read_choices(arguments.report_only)))
        return
    if arguments.out is None or arguments.manifest is None:
        raise SystemExit("--out DIR and --manifest KEY CHECKSUM are required, or --report-only DIR")
    key, checksum = arguments.manifest
    rows = choose(
        store_settings=Settings(),
        manifest=ArtifactRef(key, Checksum.parse(checksum)),
        corpora=arguments.corpora,
        workspace=arguments.workspace,
        budgets=arguments.budget or BUDGETS,
        seeds=arguments.seed or SEEDS,
        penalties=FURTHER,
        features=arguments.features,
        threads=arguments.threads,
    )
    write(rows, arguments.out)
    print(render(rows))


if __name__ == "__main__":
    main()
