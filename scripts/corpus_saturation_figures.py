"""Draw the saturation curves from the runs the report stored, never from what is in memory.

Measuring and drawing are two steps: the report writes each run's numbers as it goes and this
turns them into figures. A legend that sits on the data is then a redraw, not another evening
of training.

    uv run scripts/corpus_saturation_figures.py data/report/saturation/runs
    uv run scripts/corpus_saturation_figures.py <runs> --figures docs/verification/figures
"""

import argparse
import sys
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

import matplotlib

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A file is written, nothing is shown: the backend must not need a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.domain.saturation.saturation_verdict import GENERALISATION_GAP
from emblema.shared.kernel.compute import ComputeTier
from scripts.corpus_saturation_report import StoredRun, stored_runs

STEM = "corpus-saturation"
# A panel per experiment, wrapped so that a note-width figure stays legible past three corpora.
PANELS_PER_ROW = 3


def label_of(run: StoredRun) -> str:
    """How an experiment is named in a legend.

    Its corpus, its tier, any shape it overrode and any reading of the loss other than the
    square: two curves of one corpus that differ only in what a miss costs must not share a
    label. A run stored before the reading was recorded was scored by the square.
    """
    tier = ComputeTiers.load().profile(ComputeTier(run.settings["tier"]))
    width, _, layers, _ = run.settings["shape"].split(",")
    label = f"{run.settings['corpus']}, tier {run.settings['tier']}"
    if (int(width), int(layers)) != (tier.width, tier.layers):
        label += f", {width} wide, {layers} deep"
    if run.settings.get("loss", "mse") != "mse":
        label += f", {run.settings['loss']}, knee {run.settings['huber_delta']}"
    return label


def by_experiment(runs: Sequence[StoredRun]) -> list[tuple[str, list[StoredRun]]]:
    finished = sorted(
        (run for run in runs if run.finished), key=lambda run: (run.experiment, run.fraction)
    )
    return [(name, list(group)) for name, group in groupby(finished, key=lambda r: r.experiment)]


def draw(runs: Sequence[StoredRun], directory: Path) -> list[Path]:
    """Draw the three figures of the measurement into ``directory``; say which were written."""
    directory.mkdir(parents=True, exist_ok=True)
    grouped = by_experiment(runs)
    if not grouped:
        raise SystemExit("no finished run to draw")
    return [
        _validation(grouped, directory / f"{STEM}-validation.png"),
        _generalisation(grouped, directory / f"{STEM}-generalisation.png"),
        _epochs(grouped, directory / f"{STEM}-epochs.png"),
    ]


def _validation(grouped: Sequence[tuple[str, Sequence[StoredRun]]], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for _, runs in grouped:
        axis.plot(
            [run.fraction for run in runs],
            [run.point().relative_validation_loss for run in runs],
            marker="o",
            label=label_of(runs[0]),
        )
    axis.axhline(1.0, linestyle=":", color="grey", label="channel-mean predictor")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([run.fraction for run in grouped[0][1]])
    axis.set_xticklabels([f"{run.fraction:.0%}" for run in grouped[0][1]])
    axis.set_xlabel("share of the training units")
    axis.set_ylabel("validation loss relative to the channel-mean predictor")
    axis.set_title("Validation loss against the share of the corpus, at one budget of steps")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _generalisation(grouped: Sequence[tuple[str, Sequence[StoredRun]]], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for _, runs in grouped:
        axis.plot(
            [run.fraction for run in runs],
            [run.point().generalisation_ratio for run in runs],
            marker="s",
            label=label_of(runs[0]),
        )
    axis.axhline(GENERALISATION_GAP, linestyle="--", color="grey", label="overfitting threshold")
    axis.axhline(1.0, linestyle=":", color="grey")
    axis.set_xscale("log", base=2)
    axis.set_xticks([run.fraction for run in grouped[0][1]])
    axis.set_xticklabels([f"{run.fraction:.0%}" for run in grouped[0][1]])
    axis.set_xlabel("share of the training units")
    axis.set_ylabel("relative validation / relative training loss, last epoch")
    axis.set_title("How much worse each run does on windows it never saw")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _epochs(grouped: Sequence[tuple[str, Sequence[StoredRun]]], path: Path) -> Path:
    count = len(grouped)
    columns = min(count, PANELS_PER_ROW)
    rows = -(-count // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.8 * rows), squeeze=False)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    for axis, (_, runs) in zip(axes.flat, grouped, strict=False):
        for run in runs:
            per_epoch = int(run.settings["steps_per_epoch"])
            steps = [(row.epoch + 1) * per_epoch for row in run.epochs]
            reference = float(run.settings["validation_reference"])
            axis.plot(
                steps,
                [row.validation_loss / reference for row in run.epochs],
                label=f"{run.fraction:.0%}",
            )
        axis.set_yscale("log")
        axis.set_xlabel("optimiser steps")
        axis.set_ylabel("relative validation loss")
        axis.set_title(label_of(runs[0]), fontsize=9)
        axis.legend(fontsize=7, title="share")
    figure.suptitle("Validation loss over the run, every share at the same budget of steps")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="directory the report stores runs under")
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="directory to draw into; a figures directory beside the runs unless given",
    )
    arguments = parser.parse_args(argv)
    for path in draw(
        stored_runs(arguments.runs), arguments.figures or arguments.runs.parent / "figures"
    ):
        print(path)


if __name__ == "__main__":
    main()
