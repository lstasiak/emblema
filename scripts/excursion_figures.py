"""Draw the excursion diagnostic from the files the report stored, never from what is in memory.

Three figures: the relative loss of each backbone under the four readings, against the share of
the corpus it trained over; the relative loss on each held-out unit beside the share of the
side's squares that unit holds; and the loss by the magnitude of the hidden target, where the
squared error of the channel-mean predictor and of the model are read bin by bin.

    uv run scripts/excursion_figures.py data/report/saturation/excursions
    uv run scripts/excursion_figures.py <excursions> --figures docs/verification/figures
"""

import argparse
import sys
from collections.abc import Sequence
from itertools import groupby, pairwise
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

from scripts.excursion_report import (
    LOSSES,
    MAGNITUDES,
    MSE,
    ORDINARY,
    MagnitudeBin,
    Score,
    ScoredBackbone,
    of_loss,
    read_backbones,
    read_bins,
    read_scores,
    read_settings,
    relative,
    totals,
)

STEM = "excursion"
READINGS = {
    MSE: "every hidden token",
    ORDINARY: "within {threshold:g} SD",
    "huber": "Huber δ={delta:g}",
    "clipped": "target clipped at {threshold:g} SD",
}
# One line style per experiment, in the order the experiments are met.
STYLES = ("-", "--", ":", "-.")


def by_experiment(backbones: Sequence[ScoredBackbone]) -> list[tuple[str, list[ScoredBackbone]]]:
    ordered = sorted(backbones, key=lambda b: (b.experiment, b.fraction))
    return [(name, list(group)) for name, group in groupby(ordered, key=lambda b: b.experiment)]


def draw(directory: Path, figures: Path) -> list[Path]:
    """Draw the three figures from the files in ``directory`` into ``figures``; say which."""
    backbones = read_backbones(directory)
    if not backbones:
        raise SystemExit(f"no scored backbone in {directory}")
    settings = read_settings(directory)
    # The thresholds name the axes, so a set of files without them would be drawn with the words
    # missing rather than with a complaint.
    if not {"threshold", "huber_delta"} <= settings.keys():
        raise SystemExit(
            f"{directory} records no threshold: the figures could not say what they show"
        )
    labels = {
        loss: text.format(
            threshold=float(settings["threshold"]), delta=float(settings["huber_delta"])
        )
        for loss, text in READINGS.items()
    }
    scores, bins = read_scores(directory), read_bins(directory)
    grouped = by_experiment(backbones)
    figures.mkdir(parents=True, exist_ok=True)
    return [
        _relative_loss(grouped, scores, labels, figures / f"{STEM}-relative-loss.png"),
        _by_unit(grouped, scores, labels, figures / f"{STEM}-by-unit.png"),
        _by_magnitude(grouped, bins, figures / f"{STEM}-by-magnitude.png"),
    ]


def _relative_loss(
    grouped: Sequence[tuple[str, Sequence[ScoredBackbone]]],
    scores: Sequence[Score],
    labels: dict[str, str],
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 6))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fractions: list[float] = []
    for style, (name, backbones) in zip(STYLES, grouped, strict=False):
        fractions = sorted({backbone.fraction for backbone in backbones} | set(fractions))
        for colour, loss in zip(colours, LOSSES, strict=False):
            axis.plot(
                [backbone.fraction for backbone in backbones],
                [
                    relative(
                        [
                            row
                            for row in of_loss(scores, loss)
                            if row.experiment == name and row.fraction == backbone.fraction
                        ]
                    )
                    for backbone in backbones
                ],
                marker="o",
                linestyle=style,
                color=colour,
                label=f"{backbones[0].label}: {labels[loss]}",
            )
    axis.axhline(1.0, linestyle=":", color="grey", label="channel-mean predictor")
    axis.set_xscale("log", base=2)
    axis.set_xticks(fractions)
    axis.set_xticklabels([f"{fraction:.0%}" for fraction in fractions])
    axis.set_xlabel("share of the training units")
    axis.set_ylabel("validation loss relative to the channel-mean predictor")
    axis.set_title("The same backbones under four readings of the loss")
    # Below the axes rather than over the data: eight lines and a reference share the panel.
    axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _by_unit(
    grouped: Sequence[tuple[str, Sequence[ScoredBackbone]]],
    scores: Sequence[Score],
    labels: dict[str, str],
    path: Path,
) -> Path:
    """Each held-out unit against each share: the relative loss on it, beside the squares it holds.

    Over every hidden token in the upper row, within the threshold in the lower.
    """
    columns = len(grouped)
    figure, axes = plt.subplots(
        2, columns, figsize=(max(7.0, 4.5 * columns), 7.5), squeeze=False, sharex="col"
    )
    for column, (name, backbones) in enumerate(grouped):
        largest = max(backbone.fraction for backbone in backbones)
        reference = [
            row
            for row in of_loss(scores, MSE)
            if row.experiment == name and row.fraction == largest
        ]
        _, _, _, side_squares = totals(reference)
        by_unit: dict[str, float] = {}
        for row in reference:
            by_unit[row.unit] = by_unit.get(row.unit, 0.0) + row.trivial
        units = sorted(by_unit, key=lambda unit: (-by_unit[unit], unit))
        positions = np.arange(len(units))
        for axis, loss in zip(axes[:, column], (MSE, ORDINARY), strict=False):
            mass = axis.twinx()
            mass.bar(
                positions,
                [by_unit[unit] / side_squares if side_squares else 0.0 for unit in units],
                color="lightgrey",
                width=0.8,
            )
            mass.set_ylim(0.0, 1.0)
            mass.set_ylabel("share of the side's squares")
            mass.set_zorder(axis.get_zorder() - 1)
            axis.patch.set_visible(False)
            for backbone in backbones:
                of_share = [
                    row
                    for row in of_loss(scores, loss)
                    if row.experiment == name and row.fraction == backbone.fraction
                ]
                per_unit: dict[str, list[Score]] = {unit: [] for unit in units}
                for row in of_share:
                    per_unit[row.unit].append(row)
                axis.plot(
                    positions,
                    [relative(per_unit[unit]) for unit in units],
                    marker="o",
                    linestyle="none",
                    label=f"{backbone.fraction:.0%}",
                )
            axis.axhline(1.0, linestyle=":", color="grey")
            axis.set_yscale("log")
            axis.set_ylabel(f"relative loss, {labels[loss]}")
            axis.set_title(f"{backbones[0].label}: {labels[loss]}", fontsize=9)
            axis.legend(fontsize=7, title="share", loc="upper right")
        axes[-1, column].set_xticks(positions)
        # Whole unit keys: two units of one corpus can end in the same month or number, and a
        # tick that names two of them names neither.
        axes[-1, column].set_xticklabels(units, rotation=90, fontsize=6)
        axes[-1, column].set_xlabel("held-out unit, heaviest first")
    figure.suptitle("Relative loss on each held-out unit, against the squares the unit holds")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _by_magnitude(
    grouped: Sequence[tuple[str, Sequence[ScoredBackbone]]],
    bins: Sequence[MagnitudeBin],
    path: Path,
) -> Path:
    """Per bin of target magnitude: the mean squared error, and the model's summed error there.

    The left panel reads the channel-mean predictor's and each share's mean square bin by bin;
    the right, the share of the model's summed error each bin holds.
    """
    rows = len(grouped)
    labels = [
        f"{low:g}-{high:g}" if np.isfinite(high) else f"{low:g}+"
        for low, high in pairwise(MAGNITUDES)
    ]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(rows, 2, figsize=(9.5, 3.8 * rows), squeeze=False)
    for row, (name, backbones) in enumerate(grouped):
        mean, share = axes[row, 0], axes[row, 1]
        largest = max(backbone.fraction for backbone in backbones)
        reference = _per_bin(bins, name, largest)
        mean.plot(
            positions,
            [bin_.trivial / bin_.tokens if bin_.tokens else np.nan for bin_ in reference],
            marker="s",
            color="black",
            label="channel-mean predictor",
        )
        for backbone in backbones:
            of_share = _per_bin(bins, name, backbone.fraction)
            total = sum(bin_.model for bin_ in of_share)
            mean.plot(
                positions,
                [bin_.model / bin_.tokens if bin_.tokens else np.nan for bin_ in of_share],
                marker="o",
                label=f"{backbone.fraction:.0%}",
            )
            share.plot(
                positions,
                [bin_.model / total if total else np.nan for bin_ in of_share],
                marker="o",
                label=f"{backbone.fraction:.0%}",
            )
        for axis in (mean, share):
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, fontsize=8, rotation=45)
            axis.set_xlabel("magnitude of the hidden target, standard deviations")
            axis.legend(fontsize=7)
        mean.set_yscale("log")
        mean.set_ylabel("mean squared error")
        mean.set_title(f"{backbones[0].label}: error by magnitude", fontsize=9)
        share.set_ylabel("share of the model's summed squared error")
        share.set_title(f"{backbones[0].label}: where the loss comes from", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _per_bin(bins: Sequence[MagnitudeBin], experiment: str, fraction: float) -> list[MagnitudeBin]:
    """One backbone's bins in the order of the edges, an empty bin where none was stored."""
    stored = {
        bin_.low: bin_
        for bin_ in bins
        if bin_.experiment == experiment and bin_.fraction == fraction
    }
    return [
        stored.get(low, MagnitudeBin(experiment, fraction, low, high, 0, 0.0, 0.0))
        for low, high in pairwise(MAGNITUDES)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excursions", type=Path, help="directory the report stored its files in")
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="directory to draw into; a figures directory beside the files unless given",
    )
    arguments = parser.parse_args(argv)
    for path in draw(arguments.excursions, arguments.figures or arguments.excursions / "figures"):
        print(path)


if __name__ == "__main__":
    main()
