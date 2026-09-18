"""Measure what the training loop promises: a resumed run, the cost of a checkpoint, the precisions.

A run is interrupted inside an epoch and picked up from the checkpoint it wrote, and compared with
a run that was left alone where it matters — the losses they report and the weights they leave
behind. One such pair says nothing on a device whose arithmetic does not repeat itself, so the
runs are repeated, uninterrupted and resumed in turn, and every pair is compared: a resume adds
nothing when pairs with one lie as far apart as pairs without. Beside that: what a checkpoint costs
to fetch and to read, what an epoch costs on this machine, and which precisions this device runs.

    uv sync --all-extras
    uv run scripts/training_loop_report.py
    uv run scripts/training_loop_report.py --device mps --repeats 6
    uv run scripts/training_loop_report.py --device mps --precision fp16 --repeats 20

The properties themselves are tests (`tests/ml/test_resume_matches_uninterrupted.py` and
`tests/pretraining`); this prints the numbers a note records per machine.
"""

import argparse
import io
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import torch

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.pretraining.adapters.training.torch_precision import TorchPrecision
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.adapters.training.training_checkpoint import TrainingCheckpoint
from emblema.pretraining.domain.exceptions import UnsupportedPrecisionError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from scripts.masked_reconstruction_report import (
    DEVICES,
    Run,
    device_available,
    experiments,
    publish,
)
from scripts.reporting import dated_heading, machine, table
from scripts.vocabulary import channel_names

# Small enough to run in a minute on any machine and large enough that an epoch holds several
# optimiser steps, which is what a mid-epoch interruption needs.
DEFAULT_EXPERIMENT = "control-a-s"
DEFAULT_UNITS = 8
DEFAULT_EPOCHS = 3
CHECKPOINT_EVERY = 3
# Two of each kind is the fewest that gives a pair without a resume to hold the others against; on
# the host, where every pair is equal, it is also enough.
DEFAULT_REPEATS = 2
UNINTERRUPTED, RESUMED = "uninterrupted", "resumed"
PAIRS = ((UNINTERRUPTED, UNINTERRUPTED), (UNINTERRUPTED, RESUMED), (RESUMED, RESUMED))


@dataclass(frozen=True)
class Measured:
    """One run of the loop and what it left behind.

    Attributes:
        kind: Whether the run was left alone or interrupted and resumed.
        epochs: What each epoch measured. A resumed run reports the epoch it re-entered as the
            resumed run saw it: its validation loss is what the resume produced, and its training
            loss covers only the batches after the checkpoint, so it is never compared.
        weights: Where the run's weights were stored, which says whether two runs left the same.
        values: The weights themselves, read back from the store.
        seconds: Wall time of the whole run, the epochs a resumed run threw away included.
    """

    kind: str
    epochs: tuple[EpochOutcome, ...]
    weights: ArtifactRef
    values: dict[str, torch.Tensor]
    seconds: float


@dataclass(frozen=True)
class CheckpointCost:
    """What a checkpoint of the run costs: its size, fetching it from the store, reading it back."""

    size: int
    fetch_seconds: float
    read_seconds: float


def corpus_of(run: Run, workspace: Path) -> TrainingMixture:
    published = publish(run, workspace)
    return TrainingMixture.of(
        TrainingCorpus(
            name=run.corpus,
            checksum=published.manifest.archived.block.checksum,
            training=published.training,
            validation=published.validation,
            channels=channel_names(published.manifest.scheme.vocabulary),
        )
    )


def uninterrupted(
    configuration: ExperimentConfiguration, corpus: TrainingMixture, device: str
) -> Measured:
    store = InMemoryArtifactStore()
    started = time.perf_counter()
    epochs = tuple(TorchTrainingRuntime(store, device=device).train(configuration, corpus))
    seconds = time.perf_counter() - started
    kept = _kept(epochs[-1])
    return Measured(UNINTERRUPTED, epochs, kept, weights_of(store, kept), seconds)


def interrupted_and_resumed(
    configuration: ExperimentConfiguration, corpus: TrainingMixture, device: str
) -> tuple[Measured, CheckpointCost]:
    """Stop after the second epoch, pick the run up from its last checkpoint inside that epoch.

    Each run gets a store of its own, dropped when the run has been read: a checkpoint of the
    published tier is tens of megabytes, and a run writes several.
    """
    store = InMemoryArtifactStore()
    started = time.perf_counter()
    running: Iterator[EpochOutcome] = TorchTrainingRuntime(store, device=device).train(
        configuration, corpus
    )
    first, second = next(running), next(running)
    checkpoint = second.checkpoint
    if checkpoint is None:
        raise SystemExit("no checkpoint was written inside the second epoch; lower --every")
    resumed = tuple(
        TorchTrainingRuntime(store, device=device).train(configuration, corpus, checkpoint)
    )
    seconds = time.perf_counter() - started
    if len(resumed) != configuration.budget.epochs - 1:
        raise SystemExit(
            "the second epoch's last checkpoint fell on its boundary, so no epoch was re-entered; "
            "pick an --every that puts one inside it"
        )
    kept = _kept(resumed[-1])
    measured = Measured(RESUMED, (first, *resumed), kept, weights_of(store, kept), seconds)
    return measured, checkpoint_cost(store, checkpoint, RunSignature.of(configuration, corpus))


def weights_of(store: InMemoryArtifactStore, ref: ArtifactRef) -> dict[str, torch.Tensor]:
    return TrainedModel.read(store.get(ref)).weights


def largest_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    """The largest distance between two sets of weights, over every parameter of the model."""
    return max(
        float((value - right[key].to(value.dtype)).abs().max()) for key, value in left.items()
    )


def largest_loss_difference(left: Measured, right: Measured) -> float:
    """The largest distance between the losses of two runs, over those both measured whole.

    Every epoch's validation loss, and the last epoch's training loss: an epoch a resumed run
    re-entered is trained only in part by it, so its training loss is not the epoch's.
    """
    validation = max(
        abs(one.validation_loss - two.validation_loss)
        for one, two in zip(left.epochs, right.epochs, strict=True)
    )
    return max(validation, abs(left.epochs[-1].training_loss - right.epochs[-1].training_loss))


def pair_differences(runs: Sequence[Measured]) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """For every pair of runs, the largest weight and loss differences, grouped by kind of pair."""
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {pair: [] for pair in PAIRS}
    for left, right in combinations(runs, 2):
        pair = (left.kind, right.kind) if left.kind == UNINTERRUPTED else (right.kind, left.kind)
        grouped[pair].append(
            (largest_difference(left.values, right.values), largest_loss_difference(left, right))
        )
    return grouped


def validation_spread(runs: Sequence[Measured], index: int) -> float:
    """How far apart the runs' validation losses lie at one epoch."""
    losses = [measured.epochs[index].validation_loss for measured in runs]
    return max(losses) - min(losses)


def shared_artifacts(runs: Sequence[Measured]) -> str:
    """Whether the runs left the same weights, as the content-addressed store says."""
    pairs = list(combinations(runs, 2))
    same = sum(1 for left, right in pairs if left.weights == right.weights)
    if same == len(pairs):
        return "yes, every run"
    if same == 0:
        return "no two runs"
    return f"{same} of {len(pairs)} pairs"


def checkpoint_cost(
    store: InMemoryArtifactStore, ref: ArtifactRef, signature: RunSignature
) -> CheckpointCost:
    """What fetching a checkpoint costs — the store verifies its checksum — and reading it back."""
    started = time.perf_counter()
    content = store.get(ref)
    fetched = time.perf_counter()
    TrainingCheckpoint.read(content, signature=signature)
    return CheckpointCost(len(content), fetched - started, time.perf_counter() - fetched)


def precisions_of(device: str) -> tuple[Precision, ...]:
    """Which precisions this device runs, as the refusals are written down."""
    run = []
    for precision in Precision:
        try:
            TorchPrecision(precision, device)
        except UnsupportedPrecisionError:
            continue
        run.append(precision)
    return tuple(run)


def _kept(outcome: EpochOutcome) -> ArtifactRef:
    if outcome.backbone is None:
        raise SystemExit("the run ended without weights")
    return outcome.backbone


def pairs_table(runs: Sequence[Measured]) -> str:
    def spread(values: list[float]) -> tuple[str, str, str]:
        if not values:
            return "—", "—", "—"
        return (
            f"{min(values):.3e}",
            f"{statistics.median(values):.3e}",
            f"{max(values):.3e}",
        )

    rows = []
    for pair, differences in pair_differences(runs).items():
        weights = [weight for weight, _ in differences]
        losses = [loss for _, loss in differences]
        rows.append(
            (
                ", ".join(pair),
                str(len(differences)),
                *spread(weights),
                "—" if not losses else f"{max(losses):.3e}",
            )
        )
    return table(
        (
            "Pair",
            "Pairs",
            "Weights, smallest",
            "Weights, median",
            "Weights, largest",
            "Losses, largest",
        ),
        rows,
    )


def report(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    stated = experiments()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=sorted(stated))
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--every", type=int, default=CHECKPOINT_EVERY, help="optimiser steps between checkpoints"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="runs of each kind, uninterrupted and resumed in turn",
    )
    parser.add_argument(
        "--precision",
        choices=[str(precision) for precision in Precision],
        default=None,
        help="measure the loop at this precision rather than the experiment's",
    )
    parser.add_argument("--device", choices=DEVICES, default=None)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report" / "loop")
    arguments = parser.parse_args(argv)
    device = arguments.device or TorchTrainingRuntime(InMemoryArtifactStore()).device
    if not device_available(device):
        parser.error(f"device {device} is not available on this machine")
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    file = stated[arguments.experiment]
    precision = (
        file.configuration().precision
        if arguments.precision is None
        else Precision(arguments.precision)
    )
    runnable = precisions_of(device)
    if precision not in runnable:
        parser.error(f"{device} does not run {precision}; it runs {', '.join(map(str, runnable))}")

    configuration = replace(
        file.configuration(),
        precision=precision,
        budget=replace(file.configuration().budget, epochs=arguments.epochs),
        checkpoint=CheckpointPolicy(every_steps=arguments.every),
    )
    (corpus_name,) = file.corpora
    run = Run(configuration=configuration, corpus=corpus_name, units=arguments.units, device=device)
    corpus = corpus_of(run, arguments.workspace / run.corpus)

    runs: list[Measured] = []
    costs: list[CheckpointCost] = []
    for _ in range(arguments.repeats):
        runs.append(uninterrupted(configuration, corpus, device))
        resumed, cost = interrupted_and_resumed(configuration, corpus, device)
        runs.append(resumed)
        costs.append(cost)

    straight = next(measured for measured in runs if measured.kind == UNINTERRUPTED)
    picked_up = next(measured for measured in runs if measured.kind == RESUMED)
    counts = dict.fromkeys((UNINTERRUPTED, RESUMED), 0)
    rows = []
    for measured in runs:
        counts[measured.kind] += 1
        rows.append(
            (
                f"{measured.kind} {counts[measured.kind]}",
                str(len(measured.epochs)),
                f"{measured.epochs[-1].training_loss:.6f}",
                f"{measured.epochs[-1].validation_loss:.6f}",
                f"{measured.seconds:.1f}",
                measured.weights.checksum.digest[:12],
            )
        )
    runs_table = table(
        ("Run", "Epochs", "Last training", "Last validation", "Seconds", "Weights"), rows
    )
    epochs = table(
        ("Epoch", "Uninterrupted 1", "Resumed 1", "Largest difference, all runs", "Seconds"),
        (
            (
                str(index + 1),
                f"{one.validation_loss:.6f}",
                f"{two.validation_loss:.6f}",
                f"{validation_spread(runs, index):.3e}",
                f"{one.seconds:.1f}",
            )
            for index, (one, two) in enumerate(zip(straight.epochs, picked_up.epochs, strict=True))
        ),
    )
    size = costs[0].size
    fetch = statistics.median(cost.fetch_seconds for cost in costs)
    read = statistics.median(cost.read_seconds for cost in costs)
    facts = table(
        ("", ""),
        (
            ("Machine", machine()),
            ("Python", sys.version.split()[0]),
            ("torch", version("torch")),
            ("Device", device),
            ("Precisions this device runs", ", ".join(map(str, runnable))),
            ("Experiment", f"`experiments/{arguments.experiment}.toml`"),
            (
                "Corpus",
                f"{run.corpus} cut to {arguments.units} units: "
                f"{len(corpus.corpora[0].training)} training and "
                f"{len(corpus.corpora[0].validation)} validation windows",
            ),
            (
                "Run",
                f"{arguments.epochs} epochs, batch {configuration.budget.batch_size}, "
                f"{configuration.precision}, a checkpoint every {arguments.every} steps",
            ),
            (
                "Repeats",
                f"{arguments.repeats} uninterrupted and {arguments.repeats} resumed, in turn",
            ),
            (
                "Checkpoint",
                f"{size / 1e6:.2f} MB, fetched in {fetch * 1e3:.0f} ms, "
                f"read in {read * 1e3:.0f} ms",
            ),
            (
                "Interruption",
                "after the second epoch, resumed from its last checkpoint, which falls inside it",
            ),
            ("Same artifact", shared_artifacts(runs)),
        ),
    )
    return "\n\n".join(
        [
            f"{dated_heading()} — training loop",
            facts,
            "### Runs\n\n" + runs_table,
            "### Largest difference per pair of runs\n\n" + pairs_table(runs),
            "### Validation loss per epoch\n\n" + epochs,
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use — and ×; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(report(argv))


if __name__ == "__main__":
    main()
