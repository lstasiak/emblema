"""Which penalty MiniRocket's logistic regression chooses on a binary task, and what it costs.

Over outcomes the convolution baseline fits an L2-penalised logistic regression, its penalty
chosen by the log-loss of folds within the drawn labels, among the strengths its settings list.
A choice at the strongest strength listed says the list stops too soon. This draws the task's
budgets from its tuning side exactly as a campaign's cell does, fits the candidate by the same
code over a list of strengths that runs further, and records which one each fit chose and how
long the fit took. No window of the validation side is read: the choice is made within the
drawn labels, and nothing is scored.

The curves check a choice rather than make one: for each draw they fit a logistic regression at
every strength on the same five folds the candidate chooses over, independently of its own
cross-validation, once at the solver's default tolerance and once far tighter, and record each
strength's mean held-out log-loss beside that of answering every stay with the prevalence of its
fold's kept part. A choice the folds do not prefer, or one the solver's tolerance moves, shows.

The numbers are written to CSV first and rendered from those files.

    uv run scripts/convolution_penalty_report.py --manifest KEY CHECKSUM --out DIR [--curves]
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
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

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
from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
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
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.reporting import dated_heading, table

CHOICES = "penalty_choices.csv"
CURVES = "log_loss_curves.csv"
TASK = "physionet2012-in-hospital-death"
BUDGETS = ("50", "200", "1000", "all")
SEEDS = (1, 2, 3)
# The strengths the candidate chooses among today, a factor of about 4.64 apart, and four more
# steps of the same factor above the strongest.
IN_FORCE = (0.001, 0.00464, 0.0215, 0.1, 0.464, 2.15, 10.0, 46.4, 215.0, 1000.0)
FURTHER = (*IN_FORCE, 4640.0, 21500.0, 100000.0, 464000.0)
FEATURES = 9996
CURVE_BUDGETS = ("50", "200", "1000")
# The solver's own tolerance, and one far below any difference between two strengths' losses.
TOLERANCES = (1e-4, 1e-8)


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


@dataclass(frozen=True, kw_only=True)
class CurvePoint:
    """The folds' mean held-out log-loss of one strength on one draw, under one tolerance.

    Attributes:
        budget: The budget of labels, as the task names it.
        seed: The seed of the draw and of the convolutions.
        penalty: The strength of the L2 penalty.
        tolerance: The solver's tolerance.
        log_loss: The mean over the folds of the held-out log-loss.
        prevalence_log_loss: The same mean for answering each held-out stay with the prevalence
            of its fold's kept part.
        chosen: The strength the candidate's own fit chose on this draw.
    """

    budget: str
    seed: int
    penalty: float
    tolerance: float
    log_loss: float
    prevalence_log_loss: float
    chosen: float

    COLUMNS = (
        "budget",
        "seed",
        "penalty",
        "tolerance",
        "log_loss",
        "prevalence_log_loss",
        "chosen",
    )

    def record(self) -> dict[str, str]:
        return {column: str(getattr(self, column)) for column in self.COLUMNS}

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            budget=record["budget"],
            seed=int(record["seed"]),
            penalty=float(record["penalty"]),
            tolerance=float(record["tolerance"]),
            log_loss=float(record["log_loss"]),
            prevalence_log_loss=float(record["prevalence_log_loss"]),
            chosen=float(record["chosen"]),
        )


class Sampler:
    """The task's draws from its tuning side, and the candidate fitted on each as a cell would.

    Built once over the published corpus, so that every draw and fit of a report reads the same
    task, labels and blocks.
    """

    def __init__(
        self,
        *,
        store_settings: Settings,
        manifest: ArtifactRef,
        corpora: Path,
        workspace: Path,
        features: int,
        threads: int,
    ) -> None:
        known = KnownTasks.named(TASK)
        blocks = PublishedCorpusBlocks(configured_store(store_settings), workspace)
        corpus = BlockCorpusWindows(blocks)
        tasks = InMemoryDownstreamTaskRepository()
        self._task_id = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
            known.defined_over(manifest, corpus.describe(manifest))
        )
        task = tasks.get(self._task_id)
        self._draw = DrawLabelBudget(tasks, corpus, KnownGroundTruths.under(corpora))
        self._read = ReadCorpus.every(blocks, (task,))[task.manifest]
        self._scheme = task.label_scheme()
        self._features = features
        self._threads = threads

    def fit(
        self, budget: str, seed: int, penalties: Sequence[float]
    ) -> tuple[FittedConvolutions, NDArray[np.float64], NDArray[np.float64], float]:
        """The candidate fitted on one draw: the fit, its scaled features, targets and seconds."""
        method = RandomConvolutions(
            convolutions=MiniRocketSpec(features=self._features),
            ridge=RidgeSpec(penalties=tuple(penalties), threads=self._threads),
        )
        sample = self._draw(
            DrawLabelBudgetCommand(task=self._task_id, budget=LabelBudget.parse(budget), seed=seed)
        )
        targets = np.array([labelled.target / self._scheme.scale for labelled in sample.windows])
        windows = list(self._read.windows([labelled.window for labelled in sample.windows]))
        started = time.perf_counter()
        fitted = FittedConvolutions.fitted(
            ClassicalRecipe(method=method, seed=seed, sources=()),
            method,
            self._read.channels,
            method.convolutions.steps_over(self._read.manifest.window_length),
            windows,
            targets,
            self._scheme.scale,
            self._scheme.kind,
        )
        seconds = time.perf_counter() - started
        with threadpool_limits(limits=self._threads):
            laid = RegularGrid(fitted.grid_steps, fitted.channels).of(windows)[:, fitted.rows]
            scaled = fitted.transform.of(laid) / fitted.feature_scale
        return fitted, scaled, targets, seconds


def choose(
    sampler: Sampler, budgets: Sequence[str], seeds: Sequence[int], penalties: Sequence[float]
) -> list[PenaltyChoice]:
    """Draw every budget under every seed from the tuning side, fit, and record the choice."""
    choices = []
    for budget in budgets:
        for seed in seeds:
            fitted, _, targets, seconds = sampler.fit(budget, seed, penalties)
            choice = PenaltyChoice(
                budget=budget,
                seed=seed,
                windows=len(targets),
                positives=int(targets.sum()),
                chosen=fitted.penalty,
                strongest_in_force=max(IN_FORCE),
                seconds=seconds,
            )
            print(
                f"{budget} seed {seed}: {choice.windows} windows, penalty {choice.chosen:g}, "
                f"{choice.seconds:.0f} s",
                flush=True,
            )
            choices.append(choice)
    return choices


def curves(
    sampler: Sampler,
    budgets: Sequence[str],
    seeds: Sequence[int],
    penalties: Sequence[float],
    *,
    threads: int,
) -> list[CurvePoint]:
    """Every strength's folds' log-loss on every draw, beside the strength the fit chose."""
    points = []
    for budget in budgets:
        for seed in seeds:
            fitted, scaled, targets, _ = sampler.fit(budget, seed, penalties)
            folds = OutcomeFolds.of(targets.tolist())
            splits = [
                (list(folds.kept(fold)), list(folds.held_out(fold))) for fold in range(folds.count)
            ]
            prevalence = float(
                np.mean(
                    [
                        log_loss(
                            targets[held], np.full(len(held), targets[kept].mean()), labels=[0, 1]
                        )
                        for kept, held in splits
                    ]
                )
            )
            for penalty in penalties:
                for tolerance in TOLERANCES:
                    points.append(
                        CurvePoint(
                            budget=budget,
                            seed=seed,
                            penalty=penalty,
                            tolerance=tolerance,
                            log_loss=_folds_log_loss(
                                scaled, targets, splits, penalty, tolerance, threads
                            ),
                            prevalence_log_loss=prevalence,
                            chosen=fitted.penalty,
                        )
                    )
            print(f"{budget} seed {seed}: curve over {len(penalties)} strengths", flush=True)
    return points


def _folds_log_loss(
    scaled: NDArray[np.float64],
    targets: NDArray[np.float64],
    splits: Sequence[tuple[list[int], list[int]]],
    penalty: float,
    tolerance: float,
    threads: int,
) -> float:
    losses = []
    for kept, held in splits:
        with threadpool_limits(limits=threads):
            model = LogisticRegression(
                C=1.0 / penalty, l1_ratio=0.0, max_iter=20_000, tol=tolerance
            ).fit(scaled[kept], targets[kept])
        losses.append(
            log_loss(targets[held], model.predict_proba(scaled[held])[:, 1], labels=[0, 1])
        )
    return float(np.mean(losses))


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


def write_curves(rows: Sequence[CurvePoint], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CURVES).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CurvePoint.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.record() for row in rows)
    return directory / CURVES


def read_curves(directory: Path) -> tuple[CurvePoint, ...]:
    """The curves stored under ``directory``, or none."""
    if not (directory / CURVES).is_file():
        return ()
    with (directory / CURVES).open(newline="", encoding="utf-8") as handle:
        return tuple(CurvePoint.parse(row) for row in csv.DictReader(handle))


def render_curves(points: Sequence[CurvePoint]) -> str:
    """One row per draw: the fit's choice, the folds' best strength under each tolerance."""
    draws = list(dict.fromkeys((point.budget, point.seed) for point in points))
    rows = []
    for budget, seed in draws:
        drawn = [p for p in points if (p.budget, p.seed) == (budget, seed)]
        best = {
            tolerance: min((p for p in drawn if p.tolerance == tolerance), key=lambda p: p.log_loss)
            for tolerance in TOLERANCES
        }
        rows.append(
            (
                budget,
                str(seed),
                f"{drawn[0].chosen:g}",
                *(f"{best[t].penalty:g} ({best[t].log_loss:.4f})" for t in TOLERANCES),
                f"{drawn[0].prevalence_log_loss:.4f}",
            )
        )
    return table(
        (
            "Budget",
            "Seed",
            "Chosen by the fit",
            *(f"Folds' best at tolerance {t:g} (log-loss)" for t in TOLERANCES),
            "Prevalence alone",
        ),
        rows,
    )


def render(rows: Sequence[PenaltyChoice]) -> str:
    """The note's table: one row per fit, the grid in force marked where a choice passes it."""
    return "\n\n".join(
        [
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
        "--report-only", type=Path, help="render the tables from a directory written earlier"
    )
    parser.add_argument(
        "--curves",
        action="store_true",
        help="trace every strength's folds' log-loss instead of recording the choices",
    )
    parser.add_argument("--budget", action="append", help="a budget; every one by default")
    parser.add_argument("--seed", action="append", type=int, help="a seed; 1 to 3 by default")
    parser.add_argument("--features", type=int, default=FEATURES)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--corpora", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "workspace")
    return parser.parse_args(argv)


def report(directory: Path) -> str:
    """Every table stored under ``directory``, under one dated heading.

    Raises:
        SystemExit: If nothing is stored there.
    """
    choices = read_choices(directory) if (directory / CHOICES).is_file() else ()
    points = read_curves(directory)
    if not choices and not points:
        raise SystemExit(f"{directory} holds no {CHOICES} or {CURVES}; nothing to render")
    parts = [dated_heading()]
    if choices:
        parts.append(render(choices))
    if points:
        parts.append(render_curves(points))
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(report(arguments.report_only))
        return
    if arguments.out is None or arguments.manifest is None:
        raise SystemExit("--out DIR and --manifest KEY CHECKSUM are required, or --report-only DIR")
    key, checksum = arguments.manifest
    sampler = Sampler(
        store_settings=Settings(),
        manifest=ArtifactRef(key, Checksum.parse(checksum)),
        corpora=arguments.corpora,
        workspace=arguments.workspace,
        features=arguments.features,
        threads=arguments.threads,
    )
    seeds = arguments.seed or SEEDS
    if arguments.curves:
        points = curves(
            sampler,
            arguments.budget or CURVE_BUDGETS,
            seeds,
            FURTHER,
            threads=arguments.threads,
        )
        write_curves(points, arguments.out)
    else:
        write(choose(sampler, arguments.budget or BUDGETS, seeds, FURTHER), arguments.out)
    print(report(arguments.out))


if __name__ == "__main__":
    main()
