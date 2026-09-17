"""Measure how a corpus saturates: one budget of optimiser steps over a growing share of its units.

Each experiment in ``experiments/saturation-*.toml`` names a corpus and the budget a run over the
whole corpus gets. For every share asked for, the report derives the epochs that spend the same
optimiser steps over that share, reads the share's units out of the published corpus, trains
through the pretraining use case and records what each epoch measured as it happens, so that a
dropped session is picked up from its last checkpoint and a finished run is never trained again.
The curve of final validation losses is judged by the rules in the domain, and drawn from the
stored files by ``corpus_saturation_figures.py``.

    uv sync --all-extras
    uv run scripts/corpus_saturation_report.py --corpus esa_ad <manifest key> <checksum>
        --experiment saturation-esa_ad-s --device mps --track http://127.0.0.1:5000
    uv run scripts/corpus_saturation_report.py --report-only

Runs are stored under ``data/report/saturation/runs/<experiment>/<share>/``, one directory each,
and their checkpoints and weights under ``data/report/saturation/artifacts``. What the runs
showed is in ``docs/verification/corpus-saturation.md``.
"""

import argparse
import csv
import io
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import version
from itertools import groupby
from math import ceil
from pathlib import Path

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.cli.configured import configured_store
from emblema.entrypoints.cli.pretrain.source_revision import SourceRevision
from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.application.use_cases.pretrain_backbone import (
    PretrainBackbone,
    PretrainBackboneCommand,
)
from emblema.pretraining.domain.exceptions import (
    IncompatibleCheckpointError,
    InvalidSaturationCurveError,
)
from emblema.pretraining.domain.saturation.saturation_curve import SaturationCurve
from emblema.pretraining.domain.saturation.saturation_point import SaturationPoint
from emblema.pretraining.domain.saturation.saturation_verdict import judge
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.reporting import dated_heading, machine, table

EXPERIMENTS = REPO_ROOT / "experiments"
# The files of this measurement, told from the others by their name.
PREFIX = "saturation-"
WORKSPACE = REPO_ROOT / "data" / "report" / "saturation"
# Where the publisher leaves the blocks it wrote, so a machine that published reads them back
# without fetching them.
BLOCKS = REPO_ROOT / "data" / "artifacts"
# The shares measured: each about a doubling of the one before, up to the whole corpus.
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
RUN, EPOCHS, OUTCOME, CHECKPOINTS = "run.csv", "epochs.csv", "outcome.csv", "checkpoints.csv"
EPOCH_FIELDS = (
    "epoch",
    "training_loss",
    "validation_loss",
    "hidden_ratio",
    "seconds",
    "checkpoint_key",
    "checkpoint_checksum",
)


# ------------------------------------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------------------------------------


def experiments(directory: Path = EXPERIMENTS) -> dict[str, ExperimentFile]:
    """Every experiment of this measurement stated in ``directory``, by name."""
    stated = (ExperimentFile.load(path) for path in sorted(directory.glob(f"{PREFIX}*.toml")))
    return {file.name: file for file in stated}


def trivial_loss(windows: Sequence[TokenWindow]) -> float:
    """The error of predicting every token's channel mean, which in normalised units is zero.

    What a run's loss is read against: the mean square of the values over every token of the
    side, timeless ones included, as the masks hide any of them.
    """
    squares, count = 0.0, 0
    for window in windows:
        values = np.asarray(window.values, dtype=np.float64)
        squares += float(np.dot(values, values))
        count += values.size
    return squares / count if count else 0.0


def batches_of(corpus: TrainingCorpus, budget: TrainingBudget) -> int:
    """Micro-batches an epoch over the corpus holds, as the loader counts them."""
    return ceil(len(corpus.training) / budget.batch_size)


def equal_step_budget(base: TrainingBudget, base_batches: int, batches: int) -> TrainingBudget:
    """The budget over ``batches`` that spends the steps ``base`` spends over ``base_batches``.

    Epochs are whole, so the steps come out equal to the nearest epoch and the run records the
    number it took; the warmup keeps its share of the run rather than its count of epochs.
    """
    target = base.epochs * base.steps_per_epoch(base_batches)
    epochs = max(1, round(target / base.steps_per_epoch(batches)))
    return replace(base, epochs=epochs, warmup_epochs=base.warmup_epochs * epochs / base.epochs)


@dataclass(frozen=True)
class PlannedRun:
    """One run of the measurement: an experiment over one share of its corpus, budget derived.

    Attributes:
        experiment: The experiment the run follows.
        corpus: The corpus it reads.
        configuration: What the run does: the experiment with this share and the epochs that
            spend the experiment's steps over it.
        windows: The share's training windows and the whole validation side.
        batches: Micro-batches an epoch holds over that share.
    """

    experiment: str
    corpus: str
    configuration: ExperimentConfiguration
    windows: TrainingCorpus
    batches: int

    @property
    def fraction(self) -> float:
        return self.configuration.corpus_fraction

    @property
    def name(self) -> str:
        """What the run is called, by its directory and by the tracker alike."""
        return f"{self.fraction:g}"

    @property
    def steps(self) -> int:
        """Optimiser steps the run takes."""
        budget = self.configuration.budget
        return budget.epochs * budget.steps_per_epoch(self.batches)


def plan(
    file: ExperimentFile,
    manifest: ArtifactRef,
    reader: TrainingCorpusReader,
    fractions: Sequence[float] = FRACTIONS,
    epochs: int | None = None,
    *,
    equal_steps: bool = True,
    validation_stride: int = 1,
) -> list[PlannedRun]:
    """The runs of one experiment: one per share, each spending the whole corpus's budget.

    Args:
        file: The experiment, which states the budget for the whole corpus.
        manifest: The published corpus it reads.
        reader: What reads the corpus.
        fractions: Shares of the training units to run over.
        epochs: Epochs the whole corpus gets instead of the file's, for a shorter run that is
            recorded as such.
        equal_steps: Whether each share's epochs are derived to spend the whole corpus's steps.
            Off, every share trains the stated epochs: a run that measures what an epoch costs
            over a small share, not a point of a curve, and one the curve refuses.
        validation_stride: Score every so-many-th validation window rather than all of them —
            the same windows for every share, so the curve compares like with like. A share read
            in part trains many short epochs and scores after each; the whole validation side
            after every one of them would cost more than the training.

    Raises:
        ValueError: If the file states a share other than the whole corpus, or the stride is
            not positive.
    """
    if validation_stride < 1:
        raise ValueError(f"the validation stride must be positive, got {validation_stride}")
    base = file.configuration()
    if epochs is not None:
        base = replace(base, budget=replace(base.budget, epochs=epochs))
    if base.corpus_fraction != 1.0:
        raise ValueError(
            f"{file.name} states a share of {base.corpus_fraction:g}; an experiment of this "
            "measurement states the budget for the whole corpus"
        )
    whole = _scored_every(reader.read(manifest, base.corpus_share), validation_stride)
    base_batches = batches_of(whole, base.budget)
    runs = []
    for fraction in fractions:
        shared = replace(base, corpus_fraction=fraction)
        corpus = (
            whole
            if shared.corpus_share.is_whole
            else _scored_every(reader.read(manifest, shared.corpus_share), validation_stride)
        )
        batches = batches_of(corpus, base.budget)
        if equal_steps:
            shared = replace(shared, budget=equal_step_budget(base.budget, base_batches, batches))
        runs.append(PlannedRun(file.name, file.corpus, shared, corpus, batches))
    return runs


def _scored_every(corpus: TrainingCorpus, stride: int) -> TrainingCorpus:
    """The corpus scored on every ``stride``-th validation window.

    In the order the block holds them: the windows of every held-out unit, thinned evenly.
    """
    if stride == 1:
        return corpus
    return replace(corpus, validation=corpus.validation[::stride])


# ------------------------------------------------------------------------------------------------
# Storage: a directory per run, written as the run goes
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochRow:
    """What one epoch measured, as its line in the run's file holds it."""

    epoch: int
    training_loss: float
    validation_loss: float
    hidden_ratio: float
    seconds: float
    checkpoint: ArtifactRef | None


@dataclass(frozen=True)
class StoredRun:
    """A run as its directory holds it: what it was, what each epoch measured, whether it ended.

    Attributes:
        directory: Where the run is stored.
        settings: What the run was, key by key.
        epochs: What each epoch measured, in order.
        checkpoints: Every checkpoint the run wrote, in order, as the store logged them.
        backbone: The weights the run kept, once it reported its outcome; ``None`` before.
    """

    directory: Path
    settings: Mapping[str, str]
    epochs: tuple[EpochRow, ...]
    checkpoints: tuple[ArtifactRef, ...]
    backbone: ArtifactRef | None

    @property
    def finished(self) -> bool:
        """Whether the run reported its outcome."""
        return self.backbone is not None

    @property
    def experiment(self) -> str:
        return self.settings["experiment"]

    @property
    def fraction(self) -> float:
        return float(self.settings["corpus_fraction"])

    @property
    def steps(self) -> int:
        return int(self.settings["planned_steps"])

    @property
    def seconds(self) -> float:
        return sum(row.seconds for row in self.epochs)

    @property
    def last_checkpoint(self) -> ArtifactRef | None:
        """The last checkpoint the run wrote: where a dropped session is picked up.

        Read from the log the store keeps as checkpoints are written, so a session that drops in
        the middle of an epoch loses the steps since the last checkpoint and not the epoch; the
        epochs' own records stand in for runs stored before the log existed.
        """
        if self.checkpoints:
            return self.checkpoints[-1]
        written = [row.checkpoint for row in self.epochs if row.checkpoint is not None]
        return written[-1] if written else None

    def best_epoch(self) -> EpochRow:
        """The epoch the validation loss was lowest at: what early stopping would have kept.

        Raises:
            ValueError: If the run recorded no epoch.
        """
        if not self.epochs:
            raise ValueError(f"{self.directory} recorded no epoch")
        return min(self.epochs, key=lambda row: row.validation_loss)

    def point(self) -> SaturationPoint:
        """Where the run ended, as a point of the curve.

        Raises:
            ValueError: If the run recorded no epoch.
        """
        if not self.epochs:
            raise ValueError(f"{self.directory} recorded no epoch")
        last = self.epochs[-1]
        return SaturationPoint(
            fraction=self.fraction,
            training_loss=last.training_loss,
            validation_loss=last.validation_loss,
            training_reference=float(self.settings["training_reference"]),
            validation_reference=float(self.settings["validation_reference"]),
            steps=self.steps,
        )


def settings_of(
    run: PlannedRun, *, device: str, revision: str, validation_stride: int = 1
) -> dict[str, str]:
    """What the run is, as the file beside its epochs states it: enough to read it back alone."""
    configuration = run.configuration
    architecture, budget = configuration.architecture, configuration.budget
    return {
        "experiment": run.experiment,
        "corpus": run.corpus,
        "corpus_fraction": f"{run.fraction:g}",
        "tier": str(configuration.tier),
        "shape": (
            f"{architecture.width},{architecture.heads},{architecture.layers},"
            f"{architecture.feedforward_width}"
        ),
        "epochs": str(budget.epochs),
        "warmup_epochs": f"{budget.warmup_epochs:g}",
        "batch_size": str(budget.batch_size),
        "accumulation_steps": str(budget.accumulation_steps),
        "learning_rate": f"{budget.learning_rate:g}",
        "final_lr_fraction": f"{budget.final_lr_fraction:g}",
        "seed": str(budget.seed),
        "precision": str(configuration.precision),
        "checkpoint_every_steps": str(configuration.checkpoint.every_steps),
        "training_windows": str(len(run.windows.training)),
        "validation_windows": str(len(run.windows.validation)),
        "validation_stride": str(validation_stride),
        "training_reference": repr(trivial_loss(run.windows.training)),
        "validation_reference": repr(trivial_loss(run.windows.validation)),
        "vocabulary_size": str(run.windows.vocabulary_size),
        "block_checksum": str(run.windows.checksum),
        "steps_per_epoch": str(budget.steps_per_epoch(run.batches)),
        "planned_steps": str(run.steps),
        "device": device,
        "revision": revision,
        "machine": machine(),
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }


def write_settings(directory: Path, settings: Mapping[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / RUN).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "value"))
        writer.writerows(settings.items())


def read_run(directory: Path) -> StoredRun | None:
    """The run stored in ``directory``, or ``None`` where no run was started there."""
    if not (directory / RUN).is_file():
        return None
    settings = {row["key"]: row["value"] for row in _rows(directory / RUN)}
    return StoredRun(
        directory,
        settings,
        read_epochs(directory),
        read_checkpoints(directory),
        read_outcome(directory),
    )


def read_epochs(directory: Path) -> tuple[EpochRow, ...]:
    path = directory / EPOCHS
    if not path.is_file():
        return ()
    return tuple(
        EpochRow(
            epoch=int(row["epoch"]),
            training_loss=float(row["training_loss"]),
            validation_loss=float(row["validation_loss"]),
            hidden_ratio=float(row["hidden_ratio"]),
            seconds=float(row["seconds"]),
            checkpoint=(
                ArtifactRef(row["checkpoint_key"], Checksum.parse(row["checkpoint_checksum"]))
                if row["checkpoint_key"]
                else None
            ),
        )
        for row in _rows(path)
    )


def read_outcome(directory: Path) -> ArtifactRef | None:
    """The weights the run's outcome names, or ``None`` where the run has not ended."""
    path = directory / OUTCOME
    if not path.is_file():
        return None
    stated = {row["key"]: row["value"] for row in _rows(path)}
    return ArtifactRef(stated["backbone_key"], Checksum.parse(stated["backbone_checksum"]))


def read_checkpoints(directory: Path) -> tuple[ArtifactRef, ...]:
    """Every checkpoint the run wrote, in the order it wrote them."""
    path = directory / CHECKPOINTS
    if not path.is_file():
        return ()
    return tuple(ArtifactRef(row["key"], Checksum.parse(row["checksum"])) for row in _rows(path))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CheckpointLog:
    """A store that writes down each checkpoint a run puts, as it is put.

    The runtime reports a checkpoint only with the epoch it was written in, so a session that
    drops inside an epoch would otherwise be picked up from the epoch before. Every transient
    artifact a run stores is a checkpoint of that run, and its reference is appended to the run's
    directory the moment the store has it; the durable ones — the weights — are the outcome's to
    report. Everything else is the wrapped store's.
    """

    def __init__(self, inner: ArtifactStore, directory: Path) -> None:
        self._inner = inner
        self._directory = directory

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        ref = self._inner.put(content, retention)
        if retention is Retention.TRANSIENT:
            path = self._directory / CHECKPOINTS
            new = not path.is_file()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(("key", "checksum"))
                writer.writerow((ref.key, str(ref.checksum)))
        return ref

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self._inner.put_file(source, retention)

    def get(self, ref: ArtifactRef) -> bytes:
        return self._inner.get(ref)

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        self._inner.get_file(ref, destination)

    def exists(self, ref: ArtifactRef) -> bool:
        return self._inner.exists(ref)


class RecordingTracker:
    """Writes what a run measures into its directory as it happens, and passes it on.

    A session that drops leaves the epochs it managed and the checkpoint the last of them wrote,
    which is what the next session resumes from; a run that ends leaves its outcome, which is what
    says it need not run again. The epoch a resumed run re-enters is already on file, trained
    whole by the session that wrote it, so it is passed on and not written twice.
    """

    def __init__(self, inner: ExperimentTracker, directory: Path) -> None:
        self._inner = inner
        self._directory = directory
        self._recorded: set[int] = set()

    def begin(self, configuration: ExperimentConfiguration, *, corpus: str, run: str) -> None:
        self._inner.begin(configuration, corpus=corpus, run=run)
        self._recorded = {row.epoch for row in read_epochs(self._directory)}

    def log_epoch(self, outcome: EpochOutcome) -> None:
        self._inner.log_epoch(outcome)
        if outcome.epoch in self._recorded:
            return
        path = self._directory / EPOCHS
        new = not path.is_file()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(EPOCH_FIELDS)
            writer.writerow(
                (
                    outcome.epoch,
                    repr(outcome.training_loss),
                    repr(outcome.validation_loss),
                    repr(outcome.hidden_ratio),
                    repr(outcome.seconds),
                    "" if outcome.checkpoint is None else outcome.checkpoint.key,
                    "" if outcome.checkpoint is None else str(outcome.checkpoint.checksum),
                )
            )
        self._recorded.add(outcome.epoch)

    def end(self, outcome: TrainingOutcome) -> None:
        self._inner.end(outcome)
        with (self._directory / OUTCOME).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("key", "value"))
            writer.writerows(
                (
                    ("backbone_key", outcome.backbone.key),
                    ("backbone_checksum", str(outcome.backbone.checksum)),
                    ("epochs", str(len(outcome.epochs))),
                    ("hidden_ratio", repr(outcome.hidden_ratio)),
                )
            )


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------


def train(
    planned: PlannedRun,
    directory: Path,
    runtime_over: Callable[[ArtifactStore], TrainingRuntime],
    store: ArtifactStore,
    tracker: Callable[[], ExperimentTracker],
    *,
    device: str,
    revision: str,
    validation_stride: int = 1,
) -> StoredRun:
    """Train the run unless its directory says it ended, picking it up where it stopped.

    The runtime is built over the store with the run's checkpoint log in front of it, so that
    what the run writes is known to the directory that resumes it. A run whose checkpoint the
    runtime refuses — one that had finished when the session dropped, before its outcome was
    written — is trained again from the start, and says so.
    """
    stored = read_run(directory)
    if stored is not None and stored.finished:
        return stored
    if stored is None:
        write_settings(
            directory,
            settings_of(
                planned, device=device, revision=revision, validation_stride=validation_stride
            ),
        )
    resume = None if stored is None else stored.last_checkpoint
    command = PretrainBackboneCommand(
        configuration=planned.configuration,
        corpus=planned.windows,
        run=planned.name,
        resume_from=resume,
    )
    runtime = runtime_over(CheckpointLog(store, directory))
    try:
        PretrainBackbone(runtime, RecordingTracker(tracker(), directory))(command)
    except IncompatibleCheckpointError as error:
        if resume is None:
            raise
        print(f"{directory}: {error}; training from the start", file=sys.stderr)
        PretrainBackbone(runtime, RecordingTracker(tracker(), directory))(
            replace(command, resume_from=None)
        )
    finished = read_run(directory)
    if finished is None or not finished.finished:
        raise RuntimeError(f"{directory} did not record the run's outcome")
    return finished


def leg(
    runs: Sequence[PlannedRun],
    results: Path,
    runtime_over: Callable[[ArtifactStore], TrainingRuntime],
    store: ArtifactStore,
    tracker: Callable[[], ExperimentTracker],
    *,
    device: str,
    revision: str,
    validation_stride: int = 1,
) -> list[StoredRun]:
    """Every planned run in order, each stored under its experiment and share."""
    stored = []
    for planned in runs:
        directory = results / planned.experiment / planned.name
        started = time.perf_counter()
        finished = train(
            planned,
            directory,
            runtime_over,
            store,
            tracker,
            device=device,
            revision=revision,
            validation_stride=validation_stride,
        )
        print(
            f"{planned.experiment} share {planned.name}: {planned.configuration.budget.epochs} "
            f"epochs, {planned.steps} steps, {len(planned.windows.training)} training windows "
            f"— {time.perf_counter() - started:.0f} s this session, "
            f"{finished.seconds:.0f} s of epochs in all",
            file=sys.stderr,
        )
        stored.append(finished)
    return stored


def stored_runs(results: Path) -> list[StoredRun]:
    """Every run stored under ``results`` that recorded an epoch, by experiment and share."""
    found = [read_run(path.parent) for path in sorted(results.glob(f"*/*/{RUN}"))]
    return sorted(
        (run for run in found if run is not None and run.epochs),
        key=lambda run: (run.experiment, run.fraction),
    )


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------


def curve_of(runs: Sequence[StoredRun]) -> SaturationCurve:
    """The curve the finished runs of one experiment trace, from the smallest share up."""
    return SaturationCurve(
        tuple(run.point() for run in sorted(runs, key=lambda run: run.fraction) if run.finished)
    )


def experiment_section(name: str, runs: Sequence[StoredRun]) -> str:
    rows = []
    for run in runs:
        last, best, point = run.epochs[-1], run.best_epoch(), run.point()
        rows.append(
            (
                f"{run.fraction:.0%}",
                run.settings["training_windows"],
                run.settings["epochs"],
                run.settings["planned_steps"],
                f"{last.training_loss:.4f}",
                f"{last.validation_loss:.4f}",
                f"{point.relative_training_loss:.3f}",
                f"{point.relative_validation_loss:.3f}",
                f"{best.validation_loss / point.validation_reference:.3f} at "
                f"{best.epoch + 1}/{len(run.epochs)}",
                f"{point.generalisation_ratio:.2f}×",
                f"{run.seconds / 60:.0f} min",
                "yes" if run.finished else "**no**",
            )
        )
    references = runs[0].point()
    lines = [
        f"### {name}",
        "",
        table(
            (
                "Share",
                "Training windows",
                "Epochs",
                "Steps",
                "Last training",
                "Last validation",
                "Relative training",
                "Relative validation",
                "Best validation (epoch)",
                "Gap",
                "Minutes",
                "Finished",
            ),
            rows,
        ),
        "",
        f"Relative losses are against the trivial predictor's on the same side — the channel "
        f"mean, whose error is {references.validation_reference:.3f} on the validation side and "
        f"{references.training_reference:.3f} on the training side of the smallest share; the gap "
        "is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell "
        "at; the verdict reads the last epoch, where every share has spent its budget.",
    ]
    finished = [run for run in runs if run.finished]
    if len(finished) < 2:
        return "\n".join([*lines, "", "Fewer than two shares finished: no curve to judge yet."])
    try:
        curve = curve_of(finished)
    except InvalidSaturationCurveError as error:
        return "\n".join([*lines, "", f"These runs trace no curve: {error}."])
    gains = ", ".join(
        f"{before.fraction:.0%}→{after.fraction:.0%} {gain:+.1%}"
        for before, after, gain in zip(curve.points, curve.points[1:], curve.gains(), strict=False)
    )
    judgement = judge(curve)
    lines += [
        "",
        f"Gains in validation loss share to share: {gains}.",
        "",
        f"**{judgement.verdict.value}** — {judgement.reading}",
    ]
    return "\n".join(lines)


def render(runs: Sequence[StoredRun], *, device: str, revision: str) -> str:
    facts = table(
        ("", ""),
        (
            ("Machine", machine()),
            ("Python", sys.version.split()[0]),
            ("torch", version("torch")),
            ("Device", device),
            ("Revision", revision),
            (
                "Rule",
                "every share of a corpus spends the optimiser steps its experiment gives the "
                "whole corpus, rounded to whole epochs; every share is scored on the same "
                "validation windows",
            ),
        ),
    )
    sections = [f"{dated_heading()} — corpus saturation", facts]
    for name, of_experiment in groupby(runs, key=lambda run: run.experiment):
        sections.append(experiment_section(name, list(of_experiment)))
    return "\n\n".join(sections)


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------


def manifests_of(given: Sequence[Sequence[str]] | None) -> dict[str, ArtifactRef]:
    """The published corpora named on the command line, by corpus name."""
    return {name: ArtifactRef(key, Checksum.parse(checksum)) for name, key, checksum in given or ()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stated = experiments()
    parser.add_argument(
        "--corpus",
        nargs=3,
        action="append",
        metavar=("NAME", "KEY", "CHECKSUM"),
        help="a published corpus: its name and the reference to its manifest; repeatable",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(stated),
        help="experiment to run, in the order given; every one whose corpus is given unless named; "
        "with --report-only, the experiments to render",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(FRACTIONS),
        help="shares of the training units to run each experiment over",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs the whole corpus gets instead of the experiment's; recorded with the runs",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=available_device(),
        help="where the model trains",
    )
    parser.add_argument(
        "--track",
        default=None,
        metavar="URI",
        help="MLflow tracking server to record the runs in; kept in memory unless given",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="where runs are stored and where their checkpoints and weights go",
    )
    parser.add_argument(
        "--blocks",
        type=Path,
        default=BLOCKS,
        help="directory blocks are fetched to and mapped from; the publisher's, so a machine "
        "that published fetches nothing",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="render the runs stored so far and train nothing",
    )
    parser.add_argument(
        "--validation-stride",
        type=int,
        default=1,
        metavar="K",
        help="score every K-th validation window, the same for every share; all of them "
        "unless given",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="train every share for the stated epochs rather than the whole corpus's steps: "
        "measures what an epoch costs, and gives no curve",
    )
    arguments = parser.parse_args(argv)
    results = arguments.workspace / "runs"
    revision = SourceRevision().current()
    if arguments.report_only:
        stored = stored_runs(results)
        if arguments.experiment:
            stored = [run for run in stored if run.experiment in arguments.experiment]
        _print(render(stored, device=arguments.device, revision=revision))
        return
    manifests = manifests_of(arguments.corpus)
    names = arguments.experiment or [
        name for name, file in stated.items() if file.corpus in manifests
    ]
    for name in names:
        if stated[name].corpus not in manifests:
            parser.error(f"{name} reads corpus {stated[name].corpus!r}, which no --corpus names")
    if not names:
        parser.error("nothing to run: no experiment reads a corpus given with --corpus")
    if not device_available(arguments.device):
        parser.error(f"device {arguments.device} is not available on this machine")
    if any(not 0.0 < fraction <= 1.0 for fraction in arguments.fractions):
        parser.error("every share must lie in (0, 1]")
    if arguments.validation_stride < 1:
        parser.error("--validation-stride must be positive")

    reader = BlockTrainingCorpusReader(configured_store(Settings()), arguments.blocks)
    store = LocalDirectoryArtifactStore(arguments.workspace / "artifacts")

    def runtime_over(logged: ArtifactStore) -> TrainingRuntime:
        return TorchTrainingRuntime(logged, device=arguments.device)

    def tracker() -> ExperimentTracker:
        if arguments.track is None:
            return InMemoryExperimentTracker()
        return MlflowExperimentTracker(arguments.track)

    planned: list[PlannedRun] = []
    for name in names:
        file = stated[name]
        planned += plan(
            file,
            manifests[file.corpus],
            reader,
            sorted(arguments.fractions),
            arguments.epochs,
            equal_steps=not arguments.smoke,
            validation_stride=arguments.validation_stride,
        )
        for run in planned[-len(arguments.fractions) :]:
            print(
                f"planned {run.experiment} share {run.name}: {run.configuration.budget.epochs} "
                f"epochs of {run.batches} batches, {run.steps} steps",
                file=sys.stderr,
            )
    leg(
        planned,
        results,
        runtime_over,
        store,
        tracker,
        device=arguments.device,
        revision=revision,
        validation_stride=arguments.validation_stride,
    )
    _print(render(stored_runs(results), device=arguments.device, revision=revision))


def _print(text: str) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use — and ×; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
