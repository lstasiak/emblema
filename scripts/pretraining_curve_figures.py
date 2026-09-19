"""Draw the curve of a pretraining run from the CSV the report stored, never from the run.

One axis, one line per corpus of the mixture in the order the run read them, the mean the best
epoch is chosen by beside them, and that epoch marked: the figure the README and a note show.

    uv run scripts/pretraining_curve_figures.py data/report/pretraining/<backbone>
    uv run scripts/pretraining_curve_figures.py <dir> --figure docs/verification/figures/<name>.png
"""

import argparse
import sys
from collections.abc import Sequence
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

from scripts.pretraining_curve_report import CurveRow, corpora_of, read

STEM = "pretraining-curve"


def draw(rows: Sequence[CurveRow], path: Path) -> Path:
    """Draw the relative validation of every corpus over the epochs into ``path``."""
    corpora = corpora_of(rows)
    epochs = sorted({row.epoch for row in rows})
    relative = {(row.corpus, row.epoch): row.relative for row in rows}
    means = [sum(relative[corpus, epoch] for corpus in corpora) / len(corpora) for epoch in epochs]
    kept = next(row.epoch for row in rows if row.backbone)
    first = rows[0]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for corpus in corpora:
        axis.plot(
            [epoch + 1 for epoch in epochs],
            [relative[corpus, epoch] for epoch in epochs],
            marker="o",
            markersize=4,
            label=corpus,
        )
    axis.plot(
        [epoch + 1 for epoch in epochs],
        means,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="mean over corpora",
    )
    axis.axvline(kept + 1, linestyle=":", color="grey", linewidth=1)
    axis.text(
        kept + 1 - 0.08,
        0.98,
        "epoch kept",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color="dimgrey",
    )
    # Linear from zero: the corpora that stay at a third of the trivial predictor's loss are the
    # ones whose rise or fall the rule reads, and a log axis would flatten them into a band.
    axis.set_ylim(bottom=0)
    axis.set_xticks([epoch + 1 for epoch in epochs])
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation loss relative to the channel-mean predictor")
    axis.set_title(
        f"{first.experiment}: tier {first.tier}, {first.precision} — validation, not test",
        fontsize=10,
    )
    axis.grid(True, which="both", alpha=0.25)
    # In the gap between the corpus learnt to its floor and the rest, where no line runs.
    axis.legend(fontsize=8, loc="center", bbox_to_anchor=(0.6, 0.36), ncol=3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curve", type=Path, help="directory the report stored the curve under")
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="file to draw into; <stem>-<experiment>.png beside the curve unless given",
    )
    arguments = parser.parse_args(argv)
    rows = read(arguments.curve)
    figure = arguments.figure or arguments.curve / f"{STEM}-{rows[0].experiment}.png"
    print(draw(rows, figure))


if __name__ == "__main__":
    main()
