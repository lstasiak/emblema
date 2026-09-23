"""Score the backbones of the saturation measurement apart from the excursions of their corpus.

The satellite corpus's held-out months were read as not learnt: at every share the validation
loss stayed at or above the channel mean's, while a few tokens in a thousand — excursions past
ten standard deviations — held nine tenths of it. What the model does on the other tokens is
invisible in that number. This report scores each stored backbone on the validation windows its
run scored, under the masks its run drew, and reads the loss four ways: over every hidden token,
which reproduces the run's own validation loss; over the hidden tokens within a threshold; as a
Huber loss; and with the target clipped at the threshold. Beside that it measures, from the
block's own values and no model, how the squared magnitude of each side of the split concentrates
in tokens, windows and units, so that the figures the note quotes come from a file.

    uv sync --all-extras
    uv run scripts/excursion_report.py --corpus esa_ad <manifest key> <checksum>
        --experiment saturation-esa_ad-s --experiment saturation-esa_ad-m --device mps
    uv run scripts/excursion_report.py --report-only
    uv run scripts/excursion_figures.py data/report/saturation/excursions

Everything measured is written under ``data/report/saturation/excursions/`` as CSV — a row per
window and side of the block, a row per backbone, window and reading of the loss, a row per bin
of target magnitude — and the tables and figures are made from those files alone. One command
writes one set of files: scoring is minutes, so a second command replaces the first's files
rather than adding to them, and the experiments a set covers are named in its settings.
"""

import argparse
import csv
import io
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.config.settings import Settings
from emblema.entrypoints.cli.pretrain.source_revision import SourceRevision
from emblema.entrypoints.configured import configured_store
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.torch_precision import TorchPrecision
from emblema.pretraining.adapters.training.torch_training_runtime import VALIDATION_SEED_STRIDE
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.precision import Precision
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.corpus_saturation_figures import label_of
from scripts.corpus_saturation_report import (
    BLOCKS,
    WORKSPACE,
    StoredRun,
    corpus_of,
    experiments,
    stored_runs,
)
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.reporting import dated_heading, machine, table

# A token past this many standard deviations of its channel is an excursion: the note's own line,
# past which the satellite corpus's values are the benchmark's anomalies rather than its
# behaviour. Also where a clipped target is clipped.
THRESHOLD = 10.0
# Where the Huber loss turns linear, in normalised units: one standard deviation of the channel.
HUBER_DELTA = 1.0
# The share of windows a side's squared magnitude is read as concentrated in.
TOP_WINDOWS = 0.01
# How far the loss over every hidden token may sit from the one the run recorded before the
# scoring is called something other than the run's. Wide enough for the spread an accelerator
# gives between repeats, far narrower than a batch or a mask drawn differently.
REPRODUCTION = 1e-3
# Edges of the bins the hidden tokens are counted in by the magnitude of their target.
MAGNITUDES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, math.inf)
SETTINGS, SIDES, BACKBONES, SCORES, BINS = (
    "settings.csv",
    "sides.csv",
    "backbones.csv",
    "scores.csv",
    "bins.csv",
)
# The four readings of one prediction, and how each is described in a table.
MSE, ORDINARY, HUBER, CLIPPED = "mse", "ordinary", "huber", "clipped"
LOSSES = (MSE, ORDINARY, HUBER, CLIPPED)


# ------------------------------------------------------------------------------------------------
# The block's own values
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    """One side of a published corpus's split: its windows in block order, and each one's unit.

    The unit is kept beside the window rather than looked up, so that whoever scores a window can
    say which month or machine it came from without the block in hand.

    Attributes:
        name: Which side: ``training`` or ``validation``.
        windows: The side's windows, in the order the block holds them.
        units: The unit key of each window, in the same order.
    """

    name: str
    windows: Sequence[TokenWindow]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.units) != len(self.windows):
            raise ValueError(
                f"{len(self.units)} units for {len(self.windows)} windows of the {self.name} side"
            )

    def every(self, stride: int) -> Self:
        """The side thinned to every ``stride``-th window, as a run scored it."""
        return type(self)(self.name, self.windows[::stride], self.units[::stride])


def sides_of(manifest: PublishedCorpusManifest, block: WindowBlock) -> dict[str, Side]:
    """Both sides of the split the manifest states, read out of the block."""
    positions = {key: index for index, key in enumerate(manifest.units)}
    unit_of_window = [block.unit_of(index) for index in range(len(block))]
    sides = {}
    for name, keys in (
        ("training", manifest.training_units),
        ("validation", manifest.validation_units),
    ):
        # A unit that yielded no window is absent from the block and from the manifest's unit
        # order alike.
        chosen = {positions[key] for key in keys if key in positions}
        sides[name] = Side(
            name,
            block.of_units(chosen),
            tuple(manifest.units[unit] for unit in unit_of_window if unit in chosen),
        )
    return sides


def fetched_block(store: ArtifactStore, block: ArtifactRef, blocks: Path) -> WindowBlock:
    """The block mapped from ``blocks``, fetched there unless a copy that hashes right is present.

    The layout the publisher leaves its blocks in, under the digest, so a machine that published
    the corpus fetches nothing; a copy is hashed before it is trusted, as the training reader
    hashes it, because the directory is shared with other processes.
    """
    path = blocks / block.checksum.digest
    if not (
        path.is_file()
        and Checksum.of_chunks(chunks_of(path), block.checksum.algorithm) == block.checksum
    ):
        blocks.mkdir(parents=True, exist_ok=True)
        store.get_file(block, path)
    return WindowBlock(path)


@dataclass(frozen=True)
class WindowMass:
    """One window of a side weighed: its squared magnitude, and how much lies past the threshold.

    Attributes:
        side: Which side of the split the window is on.
        unit: The unit it was cut from.
        window: Its position among the side's windows.
        tokens: Tokens it holds.
        squares: Sum of the squares of its values: the channel-mean predictor's summed error.
        past_tokens: Tokens whose magnitude is past the threshold.
        past_squares: The sum of squares those tokens hold.
    """

    side: str
    unit: str
    window: int
    tokens: int
    squares: float
    past_tokens: int
    past_squares: float


def masses_of(side: Side, threshold: float) -> list[WindowMass]:
    """Every window of the side weighed, timeless tokens included, as the masks hide any of them."""
    rows = []
    for index, window in enumerate(side.windows):
        values = np.asarray(window.values, dtype=np.float64)
        squares = values * values
        past = np.abs(values) > threshold
        rows.append(
            WindowMass(
                side.name,
                side.units[index],
                index,
                int(values.size),
                float(squares.sum()),
                int(past.sum()),
                float(squares[past].sum()),
            )
        )
    return rows


@dataclass(frozen=True)
class Concentration:
    """How a side's squared magnitude is spread: what a mean squared error over it is decided by.

    Attributes:
        side: Which side of the split.
        windows: Windows on it.
        tokens: Tokens on it.
        mean_square: The channel-mean predictor's error over every token.
        past_tokens: Tokens past the threshold.
        past_squares_share: Share of the side's squared magnitude those tokens hold.
        top_unit: The unit holding the largest share of the squared magnitude.
        top_unit_share: That share.
        top_windows_share: Share held by the heaviest ``TOP_WINDOWS`` of the windows.
        ordinary_mean_square: The channel-mean predictor's error over the tokens within the
            threshold — the loss a model has to beat on the ordinary tokens.
    """

    side: str
    windows: int
    tokens: int
    mean_square: float
    past_tokens: int
    past_squares_share: float
    top_unit: str
    top_unit_share: float
    top_windows_share: float
    ordinary_mean_square: float

    @classmethod
    def of(cls, rows: Sequence[WindowMass], *, top_windows: float = TOP_WINDOWS) -> Self:
        """The concentration the rows of one side show.

        Raises:
            ValueError: If the rows are of no side, or of more than one.
        """
        sides = {row.side for row in rows}
        if len(sides) != 1:
            raise ValueError(f"rows of one side expected, got {sorted(sides)}")
        squares = sum(row.squares for row in rows)
        tokens = sum(row.tokens for row in rows)
        past_tokens = sum(row.past_tokens for row in rows)
        past_squares = sum(row.past_squares for row in rows)
        by_unit: dict[str, float] = {}
        for row in rows:
            by_unit[row.unit] = by_unit.get(row.unit, 0.0) + row.squares
        top_unit = max(by_unit, key=lambda unit: (by_unit[unit], unit))
        heaviest = sorted((row.squares for row in rows), reverse=True)
        count = max(1, math.ceil(top_windows * len(rows)))
        return cls(
            side=sides.pop(),
            windows=len(rows),
            tokens=tokens,
            mean_square=squares / tokens if tokens else 0.0,
            past_tokens=past_tokens,
            past_squares_share=past_squares / squares if squares else 0.0,
            top_unit=top_unit,
            top_unit_share=by_unit[top_unit] / squares if squares else 0.0,
            top_windows_share=sum(heaviest[:count]) / squares if squares else 0.0,
            ordinary_mean_square=(
                (squares - past_squares) / (tokens - past_tokens) if tokens > past_tokens else 0.0
            ),
        )


# ------------------------------------------------------------------------------------------------
# Scoring a stored backbone
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    """One backbone's loss over one window, summed, under one reading of the loss.

    Sums rather than means, so that windows combine by addition into units, sides and shares, and
    the channel-mean predictor's loss over the same tokens travels beside the model's: the ratio
    of the two is the relative loss, one where nothing was learnt.

    Attributes:
        experiment: The experiment the backbone comes from.
        fraction: The share of the corpus it trained over.
        window: The window's position among the scored validation windows.
        unit: The unit the window was cut from.
        loss: Which reading of the loss: one of ``LOSSES``.
        observed: Tokens the window holds.
        tokens: Hidden tokens this reading scored.
        model: The model's loss summed over them.
        trivial: The channel-mean predictor's loss summed over the same tokens.
    """

    experiment: str
    fraction: float
    window: int
    unit: str
    loss: str
    observed: int
    tokens: int
    model: float
    trivial: float


@dataclass(frozen=True)
class MagnitudeBin:
    """The hidden tokens of one backbone's validation pass whose target lies in one bin.

    Attributes:
        experiment: The experiment the backbone comes from.
        fraction: The share of the corpus it trained over.
        low: Lower edge of the bin, in standard deviations of the target's channel.
        high: Upper edge; infinite for the last bin.
        tokens: Hidden tokens whose target fell in it.
        model: The model's squared error summed over them.
        trivial: The sum of their squared targets.
    """

    experiment: str
    fraction: float
    low: float
    high: float
    tokens: int
    model: float
    trivial: float


def huber(error: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    """The Huber loss of each error: half its square within ``delta``, linear past it."""
    magnitude = np.abs(error)
    return np.where(magnitude <= delta, 0.5 * error * error, delta * (magnitude - 0.5 * delta))


def readings(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    threshold: float,
    delta: float,
) -> dict[str, tuple[int, float, float]]:
    """The four readings of one window's scored tokens: how many, the model's sum, the trivial sum.

    The trivial predictor is the channel mean, which in normalised units predicts zero; under the
    clipped reading it is scored against the clipped target it would be trained on.
    """
    error = predicted - target
    ordinary = np.abs(target) <= threshold
    clipped = np.clip(target, -threshold, threshold)
    return {
        MSE: (int(target.size), float(np.dot(error, error)), float(np.dot(target, target))),
        ORDINARY: (
            int(ordinary.sum()),
            float(np.dot(error[ordinary], error[ordinary])),
            float(np.dot(target[ordinary], target[ordinary])),
        ),
        HUBER: (
            int(target.size),
            float(huber(error, delta).sum()),
            float(huber(target, delta).sum()),
        ),
        CLIPPED: (
            int(target.size),
            float(np.dot(predicted - clipped, predicted - clipped)),
            float(np.dot(clipped, clipped)),
        ),
    }


def score(
    run: StoredRun,
    model: MaskedReconstruction,
    side: Side,
    masking: MaskingStrategy,
    *,
    device: str,
    threshold: float = THRESHOLD,
    delta: float = HUBER_DELTA,
) -> tuple[list[Score], list[MagnitudeBin]]:
    """Score the backbone on the side's windows exactly as its run scored them.

    The batches are the run's — its micro-batch size, in block order — and the masks of each are
    drawn from the generator the training runtime seeds for that batch, so the hidden tokens are
    the ones the run's own validation loss was read over. The precision is the run's too. What
    comes back is one row per window and reading, and the tokens binned by target magnitude.
    """
    batch_size, seed = int(run.settings["batch_size"]), int(run.settings["seed"])
    precision = TorchPrecision(Precision(run.settings["precision"]), device)
    loader = WindowLoader(side.windows, batch_size=batch_size, seed=seed, shuffle=False)
    masking_draw = TokenMasking(masking)
    model = model.to(device).eval()
    scores: list[Score] = []
    binned = np.zeros((len(MAGNITUDES) - 1, 3), dtype=np.float64)
    offset = 0
    with torch.no_grad():
        for index, on_host in enumerate(loader.batches_of(0)):
            draws = torch.Generator().manual_seed(seed * VALIDATION_SEED_STRIDE + index)
            masks = masking_draw.draw(on_host, draws)
            with precision.autocast():
                predicted = model(on_host.to(device), masks.to(device))
            # Targets and masks are read off the host batch, and the prediction is moved before
            # it is widened: a copy from the accelerator that changes the type on the way gives
            # back other numbers than went in (torch 2.14 on MPS, measured while writing this).
            scored = (masks.hidden & ~on_host.padding_mask).numpy()
            targets = on_host.features[..., 0].to(torch.float64).numpy()
            predictions = predicted.detach().to("cpu").to(torch.float64).numpy()
            observed = (~on_host.padding_mask).sum(dim=1).tolist()
            for row in range(on_host.batch_size):
                target, prediction = targets[row][scored[row]], predictions[row][scored[row]]
                window = offset + row
                for loss, (tokens, of_model, trivial) in readings(
                    target, prediction, threshold=threshold, delta=delta
                ).items():
                    scores.append(
                        Score(
                            run.experiment,
                            run.fraction,
                            window,
                            side.units[window],
                            loss,
                            int(observed[row]),
                            tokens,
                            of_model,
                            trivial,
                        )
                    )
                squared = (prediction - target) ** 2
                places = np.digitize(np.abs(target), MAGNITUDES[1:-1])
                np.add.at(binned[:, 0], places, 1)
                np.add.at(binned[:, 1], places, squared)
                np.add.at(binned[:, 2], places, target * target)
            offset += on_host.batch_size
    bins = [
        MagnitudeBin(
            run.experiment,
            run.fraction,
            MAGNITUDES[place],
            MAGNITUDES[place + 1],
            int(binned[place, 0]),
            float(binned[place, 1]),
            float(binned[place, 2]),
        )
        for place in range(len(MAGNITUDES) - 1)
    ]
    return scores, bins


def reading_the_same_block(runs: Sequence[StoredRun], block: ArtifactRef) -> None:
    """Refuse runs whose windows came out of another block than the one about to be scored.

    A backbone scored on a corpus it never read would give a table of numbers with nothing wrong
    on its face, so the manifest named on the command line is held against the block checksum
    every run recorded.

    Raises:
        ValueError: If any run read another block.
    """
    others = sorted(
        {
            run.settings.get("block_checksum", "none recorded")
            for run in runs
            if run.settings.get("block_checksum") != str(block.checksum)
        }
    )
    if others:
        raise ValueError(
            f"runs read {', '.join(others)}, not {block.checksum}: the manifest given is of "
            "another corpus, or of another publication of this one"
        )


def restored(store: ArtifactStore, run: StoredRun) -> MaskedReconstruction:
    """The backbone the run kept, rebuilt on the host.

    Raises:
        ValueError: If the run has not ended and so kept no backbone.
    """
    if run.backbone is None:
        raise ValueError(f"{run.directory} kept no backbone: the run has not ended")
    return TrainedModel.read(store.get(run.backbone)).build()


@dataclass(frozen=True)
class ScoredBackbone:
    """Which backbone was scored, and what its run had said of the same windows.

    Attributes:
        experiment: The experiment the backbone comes from.
        fraction: The share of the corpus it trained over.
        label: How the experiment is named in a figure.
        backbone: The weights scored.
        run_validation_loss: The validation loss the run's last epoch recorded, which the ``mse``
            reading over every hidden token reproduces.
        run_hidden_ratio: The share of observed tokens the run's masks hid.
        seconds: What scoring the backbone took.
    """

    experiment: str
    fraction: float
    label: str
    backbone: ArtifactRef
    run_validation_loss: float
    run_hidden_ratio: float
    seconds: float


def scored_backbone(run: StoredRun, seconds: float) -> ScoredBackbone:
    """What the run's directory says of the backbone just scored.

    Raises:
        ValueError: If the run has not ended.
    """
    if run.backbone is None or not run.epochs:
        raise ValueError(f"{run.directory} has not ended")
    last = run.epochs[-1]
    return ScoredBackbone(
        run.experiment,
        run.fraction,
        label_of(run),
        run.backbone,
        last.validation_loss,
        last.hidden_ratio,
        seconds,
    )


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------


Row = WindowMass | Score | MagnitudeBin | ScoredBackbone


def write_rows(path: Path, rows: Iterable[Row]) -> None:
    """Write the rows as CSV, a column per field; a reference as its key and its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header: tuple[str, ...] | None = None
        for row in rows:
            if header is None:
                header = tuple(_columns(row))
                writer.writerow(header)
            writer.writerow(_cells(row))


def _columns(row: Row) -> list[str]:
    names = []
    for field in fields(row):
        value = getattr(row, field.name)
        if isinstance(value, ArtifactRef):
            names += [f"{field.name}_key", f"{field.name}_checksum"]
        else:
            names.append(field.name)
    return names


def _cells(row: Row) -> list[str]:
    cells: list[str] = []
    for field in fields(row):
        value = getattr(row, field.name)
        if isinstance(value, ArtifactRef):
            cells += [value.key, str(value.checksum)]
        elif isinstance(value, float):
            cells.append(repr(value))
        else:
            cells.append(str(value))
    return cells


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_masses(directory: Path) -> list[WindowMass]:
    return [
        WindowMass(
            row["side"],
            row["unit"],
            int(row["window"]),
            int(row["tokens"]),
            float(row["squares"]),
            int(row["past_tokens"]),
            float(row["past_squares"]),
        )
        for row in read_rows(directory / SIDES)
    ]


def read_scores(directory: Path) -> list[Score]:
    return [
        Score(
            row["experiment"],
            float(row["fraction"]),
            int(row["window"]),
            row["unit"],
            row["loss"],
            int(row["observed"]),
            int(row["tokens"]),
            float(row["model"]),
            float(row["trivial"]),
        )
        for row in read_rows(directory / SCORES)
    ]


def read_bins(directory: Path) -> list[MagnitudeBin]:
    return [
        MagnitudeBin(
            row["experiment"],
            float(row["fraction"]),
            float(row["low"]),
            float(row["high"]),
            int(row["tokens"]),
            float(row["model"]),
            float(row["trivial"]),
        )
        for row in read_rows(directory / BINS)
    ]


def read_backbones(directory: Path) -> list[ScoredBackbone]:
    return [
        ScoredBackbone(
            row["experiment"],
            float(row["fraction"]),
            row["label"],
            ArtifactRef(row["backbone_key"], Checksum.parse(row["backbone_checksum"])),
            float(row["run_validation_loss"]),
            float(row["run_hidden_ratio"]),
            float(row["seconds"]),
        )
        for row in read_rows(directory / BACKBONES)
    ]


def read_settings(directory: Path) -> dict[str, str]:
    return {row["key"]: row["value"] for row in read_rows(directory / SETTINGS)}


def write_settings(directory: Path, settings: Mapping[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / SETTINGS).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "value"))
        writer.writerows(settings.items())


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------


def totals(scores: Iterable[Score]) -> tuple[int, int, float, float]:
    """Observed tokens, scored tokens, the model's sum and the trivial sum over the scores."""
    observed = tokens = 0
    model = trivial = 0.0
    for row in scores:
        observed += row.observed
        tokens += row.tokens
        model += row.model
        trivial += row.trivial
    return observed, tokens, model, trivial


def relative(scores: Iterable[Score]) -> float:
    """The model's loss over the trivial predictor's on the same tokens; ``nan`` over none."""
    _, _, model, trivial = totals(scores)
    return model / trivial if trivial else math.nan


def of_loss(scores: Iterable[Score], loss: str) -> list[Score]:
    return [row for row in scores if row.loss == loss]


def concentration_section(masses: Sequence[WindowMass], threshold: float) -> str:
    rows = []
    for side in ("training", "validation"):
        of_side = [row for row in masses if row.side == side]
        if not of_side:
            continue
        summary = Concentration.of(of_side)
        rows.append(
            (
                side,
                f"{summary.windows:,}",
                f"{summary.tokens:,}",
                f"{summary.mean_square:.3f}",
                f"{summary.past_tokens:,} ({summary.past_tokens / summary.tokens:.2%})",
                f"{summary.past_squares_share:.1%}",
                f"`{summary.top_unit}` ({summary.top_unit_share:.1%})",
                f"{summary.top_windows_share:.1%}",
                f"{summary.ordinary_mean_square:.3f}",
            )
        )
    return "\n".join(
        [
            "### Where the squared magnitude of each side lies",
            "",
            table(
                (
                    "Side",
                    "Windows",
                    "Tokens",
                    "Mean square",
                    f"Tokens past {threshold:g} SD",
                    "Their share of the squares",
                    "Unit holding most",
                    f"Heaviest {TOP_WINDOWS:.0%} of windows",
                    "Mean square within",
                ),
                rows,
            ),
            "",
            "From the block's values alone, over every token of every window of the side: the mean "
            "square is the channel-mean predictor's error there, and a squared error over the side "
            "is decided by whatever holds the squares.",
        ]
    )


def shares_section(
    name: str,
    backbones: Sequence[ScoredBackbone],
    scores: Sequence[Score],
    threshold: float,
    delta: float,
) -> str:
    rows = []
    for backbone in sorted(backbones, key=lambda b: b.fraction):
        of_share = [row for row in scores if row.fraction == backbone.fraction]
        observed, hidden, model, _ = totals(of_loss(of_share, MSE))
        rows.append(
            (
                f"{backbone.fraction:.0%}",
                f"{hidden:,}",
                f"{hidden / observed:.4f}" if observed else "—",
                f"{backbone.run_validation_loss:.4f}",
                f"{model / hidden:.4f}" if hidden else "—",
                *(f"{relative(of_loss(of_share, loss)):.3f}" for loss in LOSSES),
            )
        )
    return "\n".join(
        [
            f"### {name}",
            "",
            table(
                (
                    "Share",
                    "Hidden tokens",
                    "Hidden ratio",
                    "Validation loss, the run's",
                    "Validation loss, here",
                    "Relative, every token",
                    f"Relative, within {threshold:g} SD",
                    f"Relative, Huber δ={delta:g}",
                    f"Relative, clipped at {threshold:g} SD",
                ),
                rows,
            ),
            "",
            "Relative losses are the model's over the channel-mean predictor's on the same hidden "
            "tokens, so one is nothing learnt. Every token reproduces the run's own validation "
            "loss; within the threshold leaves the excursions out; Huber and clipped score every "
            "token under a loss that bounds what one excursion costs.",
        ]
    )


def units_section(
    name: str, backbones: Sequence[ScoredBackbone], scores: Sequence[Score], threshold: float
) -> str:
    """Each unit of the held-out side against each share: every token, then within the threshold."""
    ordered = sorted(backbones, key=lambda b: b.fraction)
    # The trivial sums are the windows' own and the same for every share of an experiment, so
    # the largest share's rows say what each unit holds.
    reference = [row for row in of_loss(scores, MSE) if row.fraction == ordered[-1].fraction]
    _, _, _, side_squares = totals(reference)
    by_unit: dict[str, list[Score]] = {}
    for row in reference:
        by_unit.setdefault(row.unit, []).append(row)
    units = sorted(by_unit, key=lambda unit: (-totals(by_unit[unit])[3], unit))
    rows = []
    for unit in units:
        _, hidden, _, squares = totals(by_unit[unit])
        cells = [
            f"`{unit}`",
            f"{hidden:,}",
            f"{squares / hidden:.3f}" if hidden else "—",
            f"{squares / side_squares:.1%}" if side_squares else "—",
        ]
        for backbone in ordered:
            of_share = [
                row for row in scores if row.unit == unit and row.fraction == backbone.fraction
            ]
            cells.append(
                f"{relative(of_loss(of_share, MSE)):.2f} / "
                f"{relative(of_loss(of_share, ORDINARY)):.2f}"
            )
        rows.append(tuple(cells))
    return "\n".join(
        [
            f"#### {name}, by held-out unit",
            "",
            table(
                (
                    "Unit",
                    "Hidden tokens",
                    "Mean square",
                    "Share of the side's squares",
                    *(f"{backbone.fraction:.0%}" for backbone in ordered),
                ),
                rows,
            ),
            "",
            "Units in order of the squares their hidden tokens hold; each share's cell is the "
            f"relative loss over every hidden token / over those within {threshold:g} SD.",
        ]
    )


def window_quantiles(scores: Sequence[Score]) -> tuple[float, float, float, float, float]:
    """How the relative loss is spread over single windows.

    The median, the 90th and 99th percentiles and the maximum of the windows' relative losses,
    and the share of the model's summed loss the heaviest ``TOP_WINDOWS`` of windows hold.
    """
    ratios = np.asarray([row.model / row.trivial for row in scores if row.trivial > 0.0])
    if ratios.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    losses = sorted((row.model for row in scores), reverse=True)
    count = max(1, math.ceil(TOP_WINDOWS * len(losses)))
    total = sum(losses)
    p50, p90, p99 = (float(q) for q in np.quantile(ratios, (0.5, 0.9, 0.99)))
    return (p50, p90, p99, float(ratios.max()), sum(losses[:count]) / total if total else math.nan)


def windows_section(
    name: str, backbones: Sequence[ScoredBackbone], scores: Sequence[Score], threshold: float
) -> str:
    rows = []
    for backbone in sorted(backbones, key=lambda b: b.fraction):
        of_share = [row for row in scores if row.fraction == backbone.fraction]
        cells = [f"{backbone.fraction:.0%}"]
        for loss in (MSE, ORDINARY):
            p50, p90, p99, top, share = window_quantiles(of_loss(of_share, loss))
            cells += [f"{p50:.2f}", f"{p90:.2f}", f"{p99:.1f}", f"{top:.1f}", f"{share:.1%}"]
        rows.append(tuple(cells))
    return "\n".join(
        [
            f"#### {name}, by window",
            "",
            table(
                (
                    "Share",
                    "Median",
                    "p90",
                    "p99",
                    "Worst",
                    f"Heaviest {TOP_WINDOWS:.0%}",
                    f"Median within {threshold:g} SD",
                    "p90",
                    "p99",
                    "Worst",
                    f"Heaviest {TOP_WINDOWS:.0%}",
                ),
                rows,
            ),
            "",
            "The relative loss of single windows, and the share of the model's summed loss the "
            f"heaviest {TOP_WINDOWS:.0%} of windows hold; over every hidden token, then within "
            "the threshold.",
        ]
    )


def render(directory: Path, *, device: str, revision: str) -> str:
    """The note's section from the files in ``directory``."""
    settings = read_settings(directory)
    threshold = float(settings.get("threshold", str(THRESHOLD)))
    delta = float(settings.get("huber_delta", str(HUBER_DELTA)))
    masses, backbones, scores = (
        read_masses(directory),
        read_backbones(directory),
        read_scores(directory),
    )
    facts = table(
        ("", ""),
        (
            ("Machine", machine()),
            ("Python", sys.version.split()[0]),
            ("torch", version("torch")),
            ("Device", device),
            ("Revision", revision),
            ("Corpus", settings.get("corpus", "—")),
            ("Block", settings.get("block_checksum", "—")),
            ("Excursion", f"a token past {threshold:g} standard deviations of its channel"),
            ("Huber δ", f"{delta:g}"),
            (
                "Rule",
                "every backbone is scored on the validation windows its run scored, under the "
                "masks its run drew; the trivial predictor is scored on the same tokens",
            ),
        ),
    )
    sections = [f"{dated_heading()} — excursions", facts]
    if masses:
        sections.append(concentration_section(masses, threshold))
    for name, group in groupby(
        sorted(backbones, key=lambda b: (b.experiment, b.fraction)), key=lambda b: b.experiment
    ):
        of_experiment = list(group)
        of_scores = [row for row in scores if row.experiment == name]
        sections += [
            shares_section(name, of_experiment, of_scores, threshold, delta),
            units_section(name, of_experiment, of_scores, threshold),
            windows_section(name, of_experiment, of_scores, threshold),
        ]
    return "\n\n".join(sections)


# ------------------------------------------------------------------------------------------------
# Measuring
# ------------------------------------------------------------------------------------------------


def measure(
    runs: Sequence[StoredRun],
    store: ArtifactStore,
    validation: Side,
    masking_of: Mapping[str, MaskingStrategy],
    directory: Path,
    *,
    device: str,
    threshold: float = THRESHOLD,
    delta: float = HUBER_DELTA,
) -> list[ScoredBackbone]:
    """Score every finished run's backbone and write the scores beside each other.

    Every run is scored on the validation side thinned as its own settings say, so runs that were
    scored on different windows are still scored here on the windows they saw. What says the
    windows and the masks were the run's is that the loss over every hidden token comes out as the
    loss the run recorded; a run where it does not is scored and reported, and said to be off.
    """
    backbones, scores, bins = [], [], []
    for run in runs:
        if not run.finished:
            print(f"{run.directory}: not finished, not scored", file=sys.stderr)
            continue
        started = time.perf_counter()
        side = validation.every(int(run.settings.get("validation_stride", 1)))
        of_run, binned = score(
            run,
            restored(store, run),
            side,
            masking_of[run.experiment],
            device=device,
            threshold=threshold,
            delta=delta,
        )
        scores += of_run
        bins += binned
        backbone = scored_backbone(run, time.perf_counter() - started)
        backbones.append(backbone)
        _, hidden, model, _ = totals(of_loss(of_run, MSE))
        reproduced = model / max(hidden, 1)
        print(
            f"{run.experiment} share {run.fraction:g}: {len(side.windows)} windows, "
            f"{hidden} hidden tokens, loss {reproduced:.4f} against the run's "
            f"{backbone.run_validation_loss:.4f}, {backbone.seconds:.0f} s",
            file=sys.stderr,
        )
        # The masking rates come from the experiment file as it stands, and the run recorded
        # neither them nor its windows; that this loss is the run's is what says they were.
        if abs(reproduced - backbone.run_validation_loss) > REPRODUCTION * max(
            backbone.run_validation_loss, 1e-12
        ):
            print(
                f"{run.directory}: the loss over every hidden token is not the run's within "
                f"{REPRODUCTION:.0%} — these windows or these masks are not the ones it was "
                "scored on",
                file=sys.stderr,
            )
        write_rows(directory / BACKBONES, backbones)
        write_rows(directory / SCORES, scores)
        write_rows(directory / BINS, bins)
    return backbones


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stated = experiments()
    parser.add_argument(
        "--corpus",
        nargs=3,
        metavar=("NAME", "KEY", "CHECKSUM"),
        help="the published corpus the backbones trained over: its name and the reference to its "
        "manifest",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(stated),
        help="experiments whose stored runs are scored; every finished run of the corpus unless "
        "named",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="standard deviations past which a token is an excursion",
    )
    parser.add_argument(
        "--huber-delta", type=float, default=HUBER_DELTA, help="where the Huber loss turns linear"
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=available_device(),
        help="where the backbones are scored",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="where the saturation runs and their weights are stored; the scores go beside them",
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
        help="render the scores stored so far and score nothing",
    )
    arguments = parser.parse_args(argv)
    directory = arguments.workspace / "excursions"
    revision = SourceRevision().current()
    if arguments.report_only:
        _print(render(directory, device=arguments.device, revision=revision))
        return
    if arguments.corpus is None:
        parser.error("--corpus is needed to score anything")
    if not device_available(arguments.device):
        parser.error(f"device {arguments.device} is not available on this machine")
    if arguments.threshold <= 0.0 or arguments.huber_delta <= 0.0:
        parser.error("--threshold and --huber-delta must be positive")
    corpus, key, checksum = arguments.corpus
    manifest_ref = ArtifactRef(key, Checksum.parse(checksum))
    runs = [
        run
        for run in stored_runs(arguments.workspace / "runs")
        if run.experiment in stated
        and corpus_of(stated[run.experiment]) == corpus
        and (not arguments.experiment or run.experiment in arguments.experiment)
    ]
    if not runs:
        parser.error(f"no stored run of an experiment over {corpus!r}")

    published_store = configured_store(Settings())
    manifest = PublishedCorpusManifestJson().decode(published_store.get(manifest_ref))
    reading_the_same_block(runs, manifest.block)
    block = fetched_block(published_store, manifest.block, arguments.blocks)
    sides = sides_of(manifest, block)
    write_settings(
        directory,
        {
            "corpus": corpus,
            "experiments": " ".join(sorted({run.experiment for run in runs})),
            "manifest_key": manifest_ref.key,
            "manifest_checksum": str(manifest_ref.checksum),
            "block_checksum": str(manifest.block.checksum),
            "threshold": f"{arguments.threshold:g}",
            "huber_delta": f"{arguments.huber_delta:g}",
            "top_windows": f"{TOP_WINDOWS:g}",
            "device": arguments.device,
            "revision": revision,
            "machine": machine(),
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        },
    )
    started = time.perf_counter()
    masses = [row for side in sides.values() for row in masses_of(side, arguments.threshold)]
    write_rows(directory / SIDES, masses)
    print(
        f"weighed {len(masses)} windows in {time.perf_counter() - started:.0f} s", file=sys.stderr
    )
    measure(
        runs,
        LocalDirectoryArtifactStore(arguments.workspace / "artifacts"),
        sides["validation"],
        {name: file.configuration().masking for name, file in stated.items()},
        directory,
        device=arguments.device,
        threshold=arguments.threshold,
        delta=arguments.huber_delta,
    )
    _print(render(directory, device=arguments.device, revision=revision))


def _print(text: str) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use — and δ; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
