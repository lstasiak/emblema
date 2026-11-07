r"""A forecasting pretext's answer for the hidden last hour against two answers with no model.

A backbone trained to fill in each channel's last hour of a window joined with its history is
asked, on windows of its own corpus, for that hour as the pretext hides it, and its decoder is
scored beside two answers that need no model: the channel's last reading before the hour,
carried forward, and the median of the channel's hourly means over the hours before. Each is
scored on every hidden cell that held a reading, by the objective's Huber loss over the cell's
five values in normalised units; a cell whose channel read nothing earlier in the window is left
out of all three, and counted.

Cells of one window share the window, and neighbouring windows share their hours, so the
intervals resample whole windows, and the decoder is compared with each answer on the same cells.
Each time axis of the corpus — a satellite's mission — is read on its own as well as together,
because a pooled mean is weighed by the axis with the most channels.

    uv run --env-file .env.r2 scripts/history_pretext_answers_report.py \\
        --result <key> <checksum> --corpus <key> <checksum> [--side validation|training] \\
        [--windows 1500] [--device mps] --out DIR

Writes ``answers-<side>.csv``, one row per scored cell, and prints the table.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the package is installed, and this file needs nothing of the other scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402

if TYPE_CHECKING:
    # The reading needs torch; scoring what it read does not, so the tests run without it.
    from emblema.pretraining.adapters.objective.pretext_reading import ReadCells

ANSWERS = ("decoder", "last_reading", "median_of_history")
COLUMNS = ("window", "axis", "channel", *ANSWERS)
VALUES = BinnedTokens.VALUE_FEATURES
LAST = BinnedTokens.FEATURE_NAMES.index("last")
MEAN = BinnedTokens.FEATURE_NAMES.index("mean")
PRESENCE = BinnedTokens.PRESENCE
RESAMPLES = 2000
CHUNK = 16


@dataclass(frozen=True)
class ScoredCell:
    """One hidden cell that held a reading, and each answer's loss on it."""

    window: int
    axis: str
    channel: int
    losses: tuple[float, float, float]


@dataclass(frozen=True)
class Comparison:
    """The answers' mean losses on one set of cells, and the decoder's paired differences."""

    label: str
    windows: int
    cells: int
    means: tuple[float, float, float]
    intervals: tuple[tuple[float, float], ...]
    differences: tuple[tuple[float, float, float], ...]


def huber(error: NDArray[np.float64], delta: float = 1.0) -> NDArray[np.float64]:
    magnitude = np.abs(error)
    return np.where(magnitude <= delta, 0.5 * error**2, delta * (magnitude - 0.5 * delta))


def scored(read: "ReadCells", *, bins: int, window: int, axis: str) -> tuple[list[ScoredCell], int]:
    """The window's hidden cells scored by each answer, and how many had no earlier reading.

    Raises:
        ValueError: If the cells are not laid bin by bin over the same channels, the draw hid
            other than the whole last bin, or the decoder gave no answers.
    """
    if read.answers is None:
        raise ValueError("the draw was asked for without the decoder's answers")
    timed = ~read.timeless
    channel_ids = read.channel_ids[timed]
    if len(channel_ids) % bins:
        raise ValueError(f"{len(channel_ids)} cells do not lie over {bins} bins")
    rows = len(channel_ids) // bins
    by_bin = channel_ids.reshape(bins, rows)
    hidden = read.hidden[timed].reshape(bins, rows)
    if (by_bin != by_bin[0]).any():
        raise ValueError("the cells are not laid bin by bin over the same channels")
    if not hidden[-1].all() or hidden[:-1].any():
        raise ValueError("the draw hid other than the whole last bin")
    full = read.full[timed].reshape(bins, rows, -1)
    answers = read.answers[timed].reshape(bins, rows, -1)
    earlier = full[:-1, :, PRESENCE] > 0
    cells: list[ScoredCell] = []
    unscored = 0
    for row in np.flatnonzero(full[-1, :, PRESENCE] > 0):
        seen = np.flatnonzero(earlier[:, row])
        if not len(seen):
            unscored += 1
            continue
        target = full[-1, row, VALUES]
        cells.append(
            ScoredCell(
                window=window,
                axis=axis,
                channel=int(by_bin[0, row]),
                losses=(
                    float(huber(answers[-1, row] - target).mean()),
                    float(huber(full[seen[-1], row, LAST] - target).mean()),
                    float(huber(np.median(full[seen, row, MEAN]) - target).mean()),
                ),
            )
        )
    return cells, unscored


def compared(
    cells: Sequence[ScoredCell], label: str, rng: np.random.Generator, resamples: int = RESAMPLES
) -> Comparison:
    """Mean losses over ``cells``, with 95 % intervals from resampling whole windows.

    Raises:
        ValueError: If no cell is given.
    """
    if not cells:
        raise ValueError(f"no cell to compare under {label!r}")
    windows = np.array([cell.window for cell in cells])
    losses = np.array([cell.losses for cell in cells])
    ids, inverse = np.unique(windows, return_inverse=True)
    sums = np.zeros((len(ids), len(ANSWERS)))
    np.add.at(sums, inverse, losses)
    counts = np.bincount(inverse).astype(np.float64)
    draws = rng.integers(0, len(ids), (resamples, len(ids)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    point = sums.sum(axis=0) / counts.sum()
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    differences = []
    for k in range(1, len(ANSWERS)):
        d_low, d_high = np.quantile(boot[:, 0] - boot[:, k], [0.025, 0.975])
        differences.append((float(point[0] - point[k]), float(d_low), float(d_high)))
    return Comparison(
        label=label,
        windows=len(ids),
        cells=len(cells),
        means=(float(point[0]), float(point[1]), float(point[2])),
        intervals=tuple((float(a), float(b)) for a, b in zip(low, high, strict=True)),
        differences=tuple(differences),
    )


def table(comparisons: Sequence[Comparison]) -> str:
    lines = [
        "| Cells | Windows | Cells scored | Decoder | Last reading | Median of history "
        "| Decoder - last reading | Decoder - median |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in comparisons:
        means = [
            f"{m:.4f} [{lo:.4f}; {hi:.4f}]"
            for m, (lo, hi) in zip(c.means, c.intervals, strict=True)
        ]
        diffs = [f"{d:+.4f} [{lo:+.4f}; {hi:+.4f}]" for d, lo, hi in c.differences]
        lines.append(f"| {c.label} | {c.windows} | {c.cells} | " + " | ".join(means + diffs) + " |")
    return "\n".join(lines)


def comparisons_of(cells: Sequence[ScoredCell], rng: np.random.Generator) -> list[Comparison]:
    """Every cell together, then each axis on its own."""
    axes = sorted({cell.axis for cell in cells})
    return [
        compared(cells, "all", rng),
        *(compared([c for c in cells if c.axis == axis], axis, rng) for axis in axes),
    ]


def write_cells(path: Path, cells: Sequence[ScoredCell]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for cell in cells:
            writer.writerow((cell.window, cell.axis, cell.channel, *cell.losses))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the store
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
        BlockTrainingCorpusReader,
    )
    from emblema.pretraining.adapters.documents.pretraining_result_json import (
        PretrainingResultJson,
    )
    from emblema.pretraining.adapters.objective.pretext_reading import PretextReading
    from emblema.pretraining.adapters.training.joined_sides import JoinedSides
    from emblema.pretraining.adapters.training.trained_model import TrainedModel
    from emblema.pretraining.domain.training.corpus_share import CorpusShare
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--corpus", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--side", choices=("validation", "training"), default="validation")
    parser.add_argument("--windows", type=int, default=1500)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    store = configured_store(Settings())
    result = PretrainingResultJson().decode(
        store.get(ArtifactRef(arguments.result[0], Checksum.parse(arguments.result[1])))
    )
    trained = TrainedModel.read(store.get(result.outcome.backbone))
    bins, history = trained.architecture.bins, trained.architecture.history
    manifest = ArtifactRef(arguments.corpus[0], Checksum.parse(arguments.corpus[1]))
    (read_shape,) = result.mixture.corpora
    # The validation side is read whole whatever the share; the training side is the run's own.
    share = (
        result.configuration.corpus_share_of(read_shape.name)
        if arguments.side == "training"
        else CorpusShare(fraction=0.01, seed=1)
    )
    corpus = BlockTrainingCorpusReader(store, arguments.workspace).read(manifest, share)
    if corpus.checksum != read_shape.checksum:
        raise SystemExit(f"{arguments.corpus[0]} is not the corpus the run read")
    training, validation = JoinedSides.of(corpus, history=history, bins=bins)
    windows, places = (
        (training, corpus.training_places)
        if arguments.side == "training"
        else (validation, corpus.validation_places)
    )
    rng = np.random.default_rng(1)
    chosen = np.sort(
        rng.choice(len(windows), size=min(arguments.windows, len(windows)), replace=False)
    )
    reading = PretextReading.of(trained, device=arguments.device)
    cells: list[ScoredCell] = []
    unscored = 0
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start : start + CHUNK].tolist()
        asked = windows.at(batch)
        drawn = reading.drawn(
            [asked[i] for i in range(len(asked))], [0] * len(asked), answered=True, tail=True
        )
        for position, read in zip(batch, drawn, strict=True):
            found, missed = scored(read, bins=bins, window=position, axis=places[position].axis)
            cells.extend(found)
            unscored += missed
    arguments.out.mkdir(parents=True, exist_ok=True)
    write_cells(arguments.out / f"answers-{arguments.side}.csv", cells)
    print(
        f"{len(chosen)} {arguments.side} windows, {len(cells)} hidden cells scored, {unscored} "
        "with no earlier reading in the window; Huber over the five values, 95 % intervals from "
        f"{RESAMPLES} resamples of whole windows"
    )
    print(table(comparisons_of(cells, rng)))


if __name__ == "__main__":  # pragma: no cover
    main()
