"""Draw a masked-reconstruction run's figures from what it stored, never from what is in memory.

Measuring and drawing are two steps: a run writes its numbers to CSV and this turns those numbers
into figures. A legend that sits on the data or a title that reads badly is then a redraw, not
another training run.

    uv run scripts/masked_reconstruction_figures.py data/report/results/control-a-20260915-101500
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A file is written, nothing is shown: the backend must not need a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.results import Results
from scripts.masked_reconstruction_assessment import (
    RunFigures,
    find_shorter,
    read,
    read_figures,
)

FIGURES = REPO_ROOT / "docs" / "verification" / "figures"


def shape_label(settings: Mapping[str, str]) -> str:
    """How the model a run trained is named under a figure: its tier and the shape it ran."""
    width, _, layers, _ = settings.get("shape", ",,,").split(",")
    return f"tier {settings.get('tier', '?')}, {width} wide, {layers} deep"


def draw_figures(
    results: Results, assessment: Assessment, figures: RunFigures, directory: Path
) -> list[Path]:
    """Draw the run's three figures into ``directory`` and say which files were written."""
    directory.mkdir(parents=True, exist_ok=True)
    corpus = results.settings.get("corpus", "run")
    stem = f"masked-reconstruction-{corpus}"
    drawn = [
        _loss(results, figures, corpus, directory / f"{stem}-loss.png"),
        _diagnostics(results, assessment, corpus, directory / f"{stem}-diagnostics.png"),
    ]
    if figures.examples:
        drawn.append(_windows(figures, corpus, directory / f"{stem}-windows.png"))
    return drawn


def _loss(results: Results, figures: RunFigures, corpus: str, path: Path) -> Path:
    curve = results.curve
    epochs = np.arange(1, len(curve.validation) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, curve.training, marker="o", label="training")
    axis.plot(epochs, curve.validation, marker="s", label="validation")
    axis.axhline(figures.interpolation_loss, linestyle="--", color="grey", label="interpolation")
    axis.axhline(figures.ridge_loss, linestyle=":", color="grey", label="ridge")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean squared error on hidden tokens")
    axis.set_title(f"{corpus}: masked reconstruction, {shape_label(results.settings)}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _windows(figures: RunFigures, corpus: str, path: Path) -> Path:
    examples = figures.examples
    figure, axes = plt.subplots(len(examples), 1, figsize=(8, 2.2 * len(examples)), sharex=True)
    for axis, example in zip(np.atleast_1d(axes), examples, strict=False):
        axis.plot(example.times, example.truth, color="black", linewidth=1, label="truth")
        axis.scatter(
            example.times[example.visible],
            example.truth[example.visible],
            s=12,
            color="black",
            label="visible",
        )
        axis.scatter(
            example.times[example.of_kind],
            example.model[example.of_kind],
            marker="x",
            color="tab:red",
            label="model",
        )
        axis.scatter(
            example.times[example.of_kind],
            example.baseline[example.of_kind],
            marker="^",
            s=14,
            color="tab:blue",
            label="matched baseline",
        )
        axis.set_ylabel(f"w{example.window} {example.channel}\n{example.kind.value}", fontsize=8)
    drawn = np.atleast_1d(axes)
    drawn[-1].set_xlabel("position in window")
    figure.suptitle(
        f"{corpus}: tokens hidden by each kind of mask, model against the matched baseline"
    )
    # Below the panels, not over the first one: the legend would sit on the data it explains,
    # and a title above it would strike through the frame.
    figure.legend(
        *drawn[0].get_legend_handles_labels(),
        fontsize=8,
        ncol=4,
        loc="lower center",
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.99))
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _diagnostics(results: Results, assessment: Assessment, corpus: str, path: Path) -> Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    judged = [summary for summary in assessment.summaries if not summary.apart]
    positions = np.arange(len(judged))
    left.bar(positions - 0.27, [s.model_error for s in judged], width=0.27, label="model")
    left.bar(positions, [s.matched_error for s in judged], width=0.27, label="matched baseline")
    left.bar(positions + 0.27, [s.linear_error for s in judged], width=0.27, label="linear")
    left.set_xticks(positions, [s.kind.value for s in judged])
    left.set_ylabel("mean squared error")
    left.set_title(f"{corpus}: triviality per kind of mask")
    left.legend()
    spectrum = results.spectrum
    frequencies = np.arange(1, len(spectrum.truth) + 1)
    right.plot(frequencies, spectrum.truth, marker="o", color="black", label="truth")
    right.plot(
        frequencies, spectrum.model_residual, marker="x", color="tab:red", label="model residual"
    )
    right.plot(
        frequencies,
        spectrum.ridge_residual,
        marker="^",
        color="tab:blue",
        label="ridge residual",
    )
    right.set_yscale("log")
    right.set_xlabel("cycles per window")
    right.set_ylabel("energy over channel-windows hidden whole")
    right.set_title("spectrum of the truth and of each residual")
    right.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def drawn_from(stored: Path, destination: Path) -> list[Path]:
    """Redraw the figures of the run stored in ``stored``, assessing it again from its numbers.

    Raises:
        SystemExit: If the directory is not a run this can draw — a run stored before the figures
            were storable holds the numbers the assessment reads and not the ones they need.
    """
    results = read(stored)
    assessment = AssessReconstructionRun(UnitBootstrap())(
        results, find_shorter(results, stored.parent)
    )
    try:
        figures = read_figures(stored)
    except (FileNotFoundError, KeyError) as missing:
        raise SystemExit(
            f"{stored} does not hold what a figure is drawn from ({missing}); it was stored "
            "before the run store kept them, and drawing it again means running it again"
        ) from missing
    return draw_figures(results, assessment, figures, destination)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="directories the report stored runs in")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    arguments = parser.parse_args(argv)
    for stored in arguments.runs:
        for path in drawn_from(stored, arguments.figures):
            print(path)


if __name__ == "__main__":
    main()
