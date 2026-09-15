"""Measure what the training loop promises: a resumed run, the cost of a checkpoint, the precisions.

A run is interrupted inside an epoch and picked up from the checkpoint it wrote, and the two runs
are compared where it matters — the losses they end on and the weights they leave behind. Beside
that: what a checkpoint costs to write and to read, what an epoch costs on this machine, and which
precisions this device will run.

    uv sync --all-extras
    uv run scripts/training_loop_report.py
    uv run scripts/training_loop_report.py --device mps --epochs 4

The properties themselves are tests (`tests/ml/test_resume_matches_uninterrupted.py` and
`tests/pretraining`); this prints the numbers a note records per machine.
"""

import argparse
import io
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version
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
from emblema.pretraining.domain.exceptions import UnsupportedPrecisionError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
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

# Small enough to run in a minute on any machine and large enough that an epoch holds several
# optimiser steps, which is what a mid-epoch interruption needs.
DEFAULT_EXPERIMENT = "control-a-s"
DEFAULT_UNITS = 8
DEFAULT_EPOCHS = 3
CHECKPOINT_EVERY = 3


@dataclass(frozen=True)
class Measured:
    """One run of the loop and what it left behind."""

    epochs: tuple[EpochOutcome, ...]
    weights: ArtifactRef
    seconds: float


def corpus_of(run: Run, workspace: Path) -> TrainingCorpus:
    published = publish(run, workspace)
    return TrainingCorpus(
        name=run.corpus,
        checksum=published.manifest.archived.block.checksum,
        training=published.training,
        validation=published.validation,
        vocabulary_size=published.vocabulary_size,
    )


def uninterrupted(
    configuration: ExperimentConfiguration,
    corpus: TrainingCorpus,
    store: InMemoryArtifactStore,
    device: str,
) -> Measured:
    started = time.perf_counter()
    epochs = tuple(TorchTrainingRuntime(store, device=device).train(configuration, corpus))
    return Measured(epochs, _kept(epochs[-1]), time.perf_counter() - started)


def interrupted_and_resumed(
    configuration: ExperimentConfiguration,
    corpus: TrainingCorpus,
    store: InMemoryArtifactStore,
    device: str,
) -> tuple[Measured, ArtifactRef]:
    """Stop inside the second epoch, pick the run up from its checkpoint, and report the rest."""
    started = time.perf_counter()
    running: Iterator[EpochOutcome] = TorchTrainingRuntime(store, device=device).train(
        configuration, corpus
    )
    stopped = [next(running), next(running)]
    checkpoint = stopped[-1].checkpoint
    if checkpoint is None:
        raise SystemExit("no checkpoint was written inside the second epoch; lower --every")
    resumed = tuple(
        TorchTrainingRuntime(store, device=device).train(configuration, corpus, checkpoint)
    )
    return (
        Measured((*stopped, *resumed[1:]), _kept(resumed[-1]), time.perf_counter() - started),
        checkpoint,
    )


def weights_of(store: InMemoryArtifactStore, ref: ArtifactRef) -> dict[str, torch.Tensor]:
    return TrainedModel.read(store.get(ref)).weights


def largest_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    """The largest distance between two sets of weights, over every parameter of the model."""
    return max(
        float((value - right[key].to(value.dtype)).abs().max()) for key, value in left.items()
    )


def checkpoint_cost(store: InMemoryArtifactStore, ref: ArtifactRef) -> tuple[int, float]:
    """How large a checkpoint is and how long it takes to read back, in bytes and seconds."""
    started = time.perf_counter()
    content = store.get(ref)
    return len(content), time.perf_counter() - started


def precisions_of(device: str) -> str:
    """Which precisions this device runs, as the refusals are written down."""
    run = []
    for precision in Precision:
        try:
            TorchPrecision(precision, device)
        except UnsupportedPrecisionError:
            continue
        run.append(str(precision))
    return ", ".join(run) or "none"


def _kept(outcome: EpochOutcome) -> ArtifactRef:
    if outcome.backbone is None:
        raise SystemExit("the run ended without weights")
    return outcome.backbone


def report(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    stated = experiments()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=sorted(stated))
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--every", type=int, default=CHECKPOINT_EVERY, help="optimiser steps between checkpoints"
    )
    parser.add_argument("--device", choices=DEVICES, default=None)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report" / "loop")
    arguments = parser.parse_args(argv)
    device = arguments.device or TorchTrainingRuntime(InMemoryArtifactStore()).device
    if not device_available(device):
        parser.error(f"device {device} is not available on this machine")

    file = stated[arguments.experiment]
    configuration = replace(
        file.configuration(),
        budget=replace(file.configuration().budget, epochs=arguments.epochs),
        checkpoint=CheckpointPolicy(every_steps=arguments.every),
    )
    run = Run(configuration=configuration, corpus=file.corpus, units=arguments.units, device=device)
    corpus = corpus_of(run, arguments.workspace / run.corpus)

    whole, dropped = InMemoryArtifactStore(), InMemoryArtifactStore()
    straight = uninterrupted(configuration, corpus, whole, device)
    picked_up, checkpoint = interrupted_and_resumed(configuration, corpus, dropped, device)
    size, read_seconds = checkpoint_cost(dropped, checkpoint)
    difference = largest_difference(
        weights_of(whole, straight.weights), weights_of(dropped, picked_up.weights)
    )

    runs = table(
        ("Run", "Epochs", "Last training", "Last validation", "Seconds", "Weights"),
        (
            (
                name,
                str(len(measured.epochs)),
                f"{measured.epochs[-1].training_loss:.6f}",
                f"{measured.epochs[-1].validation_loss:.6f}",
                f"{measured.seconds:.1f}",
                measured.weights.checksum.digest[:12],
            )
            for name, measured in (("uninterrupted", straight), ("resumed", picked_up))
        ),
    )
    epochs = table(
        ("Epoch", "Uninterrupted", "Resumed", "Seconds"),
        (
            (
                str(index + 1),
                f"{one.validation_loss:.6f}",
                f"{two.validation_loss:.6f}",
                f"{one.seconds:.1f}",
            )
            for index, (one, two) in enumerate(zip(straight.epochs, picked_up.epochs, strict=True))
        ),
    )
    facts = table(
        ("", ""),
        (
            ("Machine", machine()),
            ("Python", sys.version.split()[0]),
            ("torch", version("torch")),
            ("Device", device),
            ("Precisions this device runs", precisions_of(device)),
            ("Experiment", f"`experiments/{arguments.experiment}.toml`"),
            (
                "Corpus",
                f"{run.corpus} cut to {arguments.units} units: "
                f"{len(corpus.training)} training and {len(corpus.validation)} validation windows",
            ),
            (
                "Run",
                f"{arguments.epochs} epochs, batch {configuration.budget.batch_size}, "
                f"{configuration.precision}, a checkpoint every {arguments.every} steps",
            ),
            ("Checkpoint", f"{size / 1e6:.2f} MB, read in {read_seconds * 1e3:.0f} ms"),
            (
                "Interruption",
                "after the second epoch's checkpoint, which falls inside the epoch",
            ),
            ("Largest weight difference", f"{difference:.3e}"),
            (
                "Same artifact",
                "yes" if straight.weights == picked_up.weights else "no",
            ),
        ),
    )
    return "\n\n".join(
        [
            f"{dated_heading()} — training loop",
            facts,
            "### Runs\n\n" + runs,
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
