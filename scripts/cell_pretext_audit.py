"""Can what a hidden cell held be answered without a model, corpus by corpus?

The audit a pretext over cells passes before any accelerator trains on it, read apart for each
corpus a pretraining mixes, since on a corpus read on a grid every cell holds a reading and the
pretext is another question than on the intensive-care stays. A window is laid out as cells of
channel × time bin, the masks of an experiment file are drawn over the full cells, and the
window is laid out again from the readings outside the hidden cells, as the encoder reads it.
For every hidden timed cell the targets are what the full layout holds there: whether anything
was measured (``presence``), ``log_count``, and the ``mean``, ``minimum`` and ``maximum`` of
what was, the last three where something was. Each is answered from the visible layout by
answers that learn nothing from the scored windows:

- ``carry``: what the visible layout carries into the cell — the channel's last visible value
  for the values, the channel's share of visible cells that held a reading for the presence,
  smoothed by one imagined cell of either kind as the objective's trivial predictor is, its
  visible mean count for the count;
- ``interpolation``: a line through the channel's nearest visible readings either side of the
  bin, for the values; the carry for the rest;
- ``cross``: a ridge map, per channel, from the other channels' visible presence and value in
  the same bin and from the bin itself; a map's answer for the presence is clipped to
  ``[0.01, 0.99]``, since a line through a 0/1 target overshoots;
- ``schedule``: a ridge map, per channel, from the channel's visible presence and counts over
  the window's bins, which bins are hidden and the bin itself — no value enters;
- ``stacked``: a ridge map over all of the above, the strongest linear answer without a model.

With ``--model`` the backbone, pretrained over cells, answers too (``model``): its decoder's
prediction for the same hidden cells under the same masks, the presence's logit read as a
probability. ``--model trained`` reads the weights kept, ``--model initialisation`` the objective
as the run drew it before a step, so what training added is read on the same cells. A corpus the
backbone never read is taken from another backbone's inputs (``--windows-of``); the rows of its
channels, which no pretraining set, are drawn or set to the mean of the learnt rows.

A backbone that read only part of a corpus's training units leaves the rest unread: neither its
training nor the validation that kept its epoch saw them. With ``--unread-units`` the answers are
scored on those units' windows and the maps fitted on the units it read, so the model's column
carries no optimism from the epoch chosen on the validation side.

The ridge maps are fitted on one half of the training windows, their penalty chosen from a grid
by the squared error on the other half, where the stack is then fitted, and every answer is
scored on the validation windows, so nothing is scored where it was fitted. A
value target is scored by the squared error against the channel's mean, zero in normalised
units; the presence by the log loss against the training windows' share of hidden cells that
held a reading. Scores are kept as sums per corpus, kind of mask, target and answer, so the R²
of any grouping is computed from the CSV without the corpora at hand; the tables are rendered
from it.

    uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone <id>
        --masks experiments/<file>.toml [--corpus <name>] [--out DIR]
    uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone <id over cells>
        --model trained [--masks experiments/<file>.toml] [--device mps]
        [--corpus <name> --windows-of <id> --unset-rows mean]
    uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone <id over cells>
        --model trained --corpus <name read in part> --unread-units
    uv run [--env-file .env.r2] scripts/cell_pretext_audit.py --manifest <key> <checksum>
        --corpus <name> --masks experiments/<file>.toml
    uv run scripts/cell_pretext_audit.py --report-only data/report/cell-pretext/<run>

A corpus no backbone has read yet is read from its published manifest (``--manifest``), its
windows as the publication cut them; only the answers without a model are scored then.
"""

import argparse
import csv
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.objective.cell_reconstruction import CellReconstruction
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.training.cell_batches import CellBatches
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.corpus_saturation_report import BLOCKS
from scripts.excursion_report import Side, fetched_block, sides_of
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.pretext_triviality_report import (
    BATCH_SIZE,
    FIT_SEED,
    TRAINING_WINDOWS,
    VALIDATION_SEED,
    VALIDATION_WINDOWS,
    CorpusWindows,
    _backbone_id,
    _registry,
    channels_apart_of,
    corpora_of,
    evenly_spaced,
    grown,
    own_channels,
    supplied_by,
    unset_rows_to_mean,
)
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "cell-pretext"
TARGETS = ("presence", "log_count", "mean", "minimum", "maximum")
ANSWERS = ("carry", "interpolation", "cross", "schedule", "stacked")
MODEL = "model"
MODES = ("trained", "initialisation")
OWN_MASKS = "the backbone's own"
SCORED_ON = {False: "the validation side", True: "the training units the run left unread"}
KINDS = (*(kind.value for kind in MaskKind), "all")
SCORES, CELLS, SETTINGS = "scores.csv", "cells.csv", "settings.txt"
PRESENCE, COUNT = BinnedTokens.PRESENCE, BinnedTokens.LOG_COUNT
MEAN, MINIMUM, MAXIMUM, FIRST, LAST = range(
    BinnedTokens.VALUE_FEATURES.start, BinnedTokens.VALUE_FEATURES.stop
)
TARGET_FEATURES = (PRESENCE, COUNT, MEAN, MINIMUM, MAXIMUM)
# A probability without a model is kept off the ends, where its log loss is infinite; a ridge
# map's answer for the presence further off them, since a line through a 0/1 target overshoots.
EDGE = 1e-6
RIDGE_EDGE = 0.01
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
STACK_PENALTY = 1.0
SMOOTHING = 1.0


@dataclass
class HiddenCells:
    """Every hidden timed cell of some windows: its targets, and what is visible around it.

    Rows are cells; the columns of ``targets`` are ``TARGETS`` in order. The designs are what
    the ridge maps read: ``cross`` the other channels' visible presence and value in the same
    bin and the bin, ``schedule`` the channel's own visible presence and counts over the bins,
    which of them are hidden, and the bin. ``carry`` and ``interpolation`` are answers already,
    and so is ``model`` where the grids carried a model's prediction.
    """

    channel: list[int] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    targets: list[NDArray[np.float64]] = field(default_factory=list)
    carry: list[NDArray[np.float64]] = field(default_factory=list)
    interpolation: list[NDArray[np.float64]] = field(default_factory=list)
    cross: list[NDArray[np.float64]] = field(default_factory=list)
    schedule: list[NDArray[np.float64]] = field(default_factory=list)
    model: list[NDArray[np.float64]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.channel)

    def array(self, name: str) -> NDArray[np.float64]:
        rows = getattr(self, name)
        return np.stack(rows) if rows else np.zeros((0, len(TARGETS)))


@dataclass(frozen=True)
class Grid:
    """One window as the grids an audit reads: full, visible, hidden and the kind of each cell.

    ``predicted`` is a model's answer for every cell, the decoder's outputs in the order a cell's
    features stand, where a model was asked; ``None`` otherwise.
    """

    rows: NDArray[np.int64]
    full: NDArray[np.float64]
    visible: NDArray[np.float64]
    hidden: NDArray[np.bool_]
    kind: NDArray[np.str_]
    predicted: NDArray[np.float64] | None = None


def grids_of(
    windows: Sequence[TokenWindow],
    layout: BinnedTokens,
    strategy: MaskingStrategy,
    *,
    seed: int,
    model: CellReconstruction | None = None,
    device: str = "cpu",
) -> list[Grid]:
    """Each window as its grids, under the masks the strategy draws from ``seed``, batch by batch.

    The timeless cells are left aside: a static feature has no span to be hidden over. A
    ``model`` is fed each batch on ``device`` under the masks drawn on the host, so its answers
    stand cell for cell beside the answers without a model; they come back to the host before
    they are widened.
    """
    batches = CellBatches(layout)
    if model is not None:
        model = model.to(device).eval()
    grids = []
    for start in range(0, len(windows), BATCH_SIZE):
        chunk = windows[start : start + BATCH_SIZE]
        collated = batches.collate(chunk)
        masks = TokenMasking(strategy).draw(collated, torch.Generator().manual_seed(seed + start))
        scored = batches.scored(collated, masks)
        prediction = None
        if model is not None:
            with torch.no_grad():
                on_host = model(scored.to(device), masks.to(device)).detach().to("cpu")
            prediction = on_host.to(torch.float64).numpy()
        # Every kind, drawn or not: a channel can be left hidden whole by the other draws.
        kinds = {kind.value: masks.of_kind(kind).numpy() for kind in MaskKind}
        for row in range(len(chunk)):
            cells = int((~collated.padding_mask[row]).sum())
            statics = int(collated.timeless[row, :cells].sum())
            ids = collated.channel_ids[row, statics:cells].numpy()
            bins = layout.bins
            rows = ids[: (cells - statics) // bins]
            shape = (bins, len(rows))
            full = scored.features[row, statics:cells].numpy().astype(np.float64)
            visible = scored.visible[row, statics:cells].numpy().astype(np.float64)
            hidden = masks.hidden[row, statics:cells].numpy()
            kind = np.full(cells - statics, "", dtype="<U8")
            for name, of_kind in kinds.items():
                kind[of_kind[row, statics:cells]] = name
            grids.append(
                Grid(
                    rows=rows,
                    full=full.reshape(*shape, -1).transpose(1, 0, 2),
                    visible=visible.reshape(*shape, -1).transpose(1, 0, 2),
                    hidden=hidden.reshape(shape).T,
                    kind=kind.reshape(shape).T,
                    predicted=(
                        None
                        if prediction is None
                        else prediction[row, statics:cells].reshape(*shape, -1).transpose(1, 0, 2)
                    ),
                )
            )
    return grids


def hidden_cells_of(grids: Iterable[Grid], *, vocabulary_size: int, bins: int) -> HiddenCells:
    """The hidden timed cells of ``grids``, with their targets, answers and designs."""
    cells = HiddenCells()
    for grid in grids:
        visible_cells = ~grid.hidden
        presence, count = grid.visible[..., PRESENCE], grid.visible[..., COUNT]
        held = visible_cells & (presence > 0.5)
        window_presence = _smoothed(presence[visible_cells]) if visible_cells.any() else 0.5
        window_count = count[held].mean() if held.any() else 0.0
        occupancy = np.zeros((vocabulary_size + 1, bins))
        level = np.zeros((vocabulary_size + 1, bins))
        occupancy[grid.rows] = np.where(visible_cells, presence, 0.0)
        level[grid.rows] = np.where(held, grid.visible[..., MEAN], 0.0)
        for r, channel in enumerate(grid.rows):
            own_visible = visible_cells[r]
            own_held = held[r]
            rate = _smoothed(presence[r][own_visible]) if own_visible.any() else window_presence
            counts = count[r][own_held].mean() if own_held.any() else window_count
            for b in np.flatnonzero(grid.hidden[r]):
                target = grid.full[r, b, list(TARGET_FEATURES)]
                carried = grid.visible[r, b, LAST]
                cells.channel.append(int(channel))
                cells.kind.append(str(grid.kind[r, b]))
                cells.targets.append(target)
                cells.carry.append(np.array([rate, counts, carried, carried, carried]))
                between = _interpolated(grid.visible[r], own_held, int(b), carried)
                cells.interpolation.append(np.array([rate, counts, between, between, between]))
                if grid.predicted is not None:
                    answer = grid.predicted[r, b, list(TARGET_FEATURES)].copy()
                    # The logistic through tanh, which no logit overflows.
                    answer[0] = 0.5 * (1.0 + math.tanh(0.5 * answer[0]))
                    cells.model.append(answer)
                others = np.ones(vocabulary_size + 1, dtype=bool)
                others[channel] = False
                cells.cross.append(
                    np.concatenate((occupancy[others, b], level[others, b], np.eye(bins)[b]))
                )
                cells.schedule.append(
                    np.concatenate(
                        (
                            np.where(own_visible, presence[r], 0.0),
                            np.where(own_held, count[r], 0.0),
                            grid.hidden[r].astype(np.float64),
                            np.eye(bins)[b],
                        )
                    )
                )
    return cells


def _smoothed(shares: NDArray[np.float64]) -> float:
    """A share read off a few cells, with one imagined cell of either kind added."""
    return float((shares.sum() + SMOOTHING * 0.5 * 2) / (shares.size + 2 * SMOOTHING))


def _interpolated(
    row: NDArray[np.float64], held: NDArray[np.bool_], bin_index: int, carried: float
) -> float:
    """A line through the channel's nearest visible readings either side of the bin."""
    before = np.flatnonzero(held[:bin_index])
    after = np.flatnonzero(held[bin_index + 1 :]) + bin_index + 1
    if before.size and after.size:
        b0, b1 = int(before[-1]), int(after[0])
        y0, y1 = row[b0, LAST], row[b1, FIRST]
        return float(y0 + (y1 - y0) * (bin_index - b0) / (b1 - b0))
    if after.size:
        return float(row[int(after[0]), FIRST])
    return carried


class PerChannelRidge:
    """A ridge map per channel from a design to a target, falling back to the fitted mean."""

    def __init__(self, penalty: float = STACK_PENALTY) -> None:
        self._penalty = penalty
        self._weights: dict[int, NDArray[np.float64]] = {}
        self._means: dict[int, float] = {}
        self._overall = 0.0

    def fit(
        self,
        channel: Sequence[int],
        design: NDArray[np.float64],
        target: NDArray[np.float64],
        defined: NDArray[np.bool_],
    ) -> "PerChannelRidge":
        channels = np.asarray(channel)
        if defined.any():
            self._overall = float(target[defined].mean())
        for which in np.unique(channels):
            rows = (channels == which) & defined
            if rows.sum() < 2:
                continue
            x = np.column_stack((design[rows], np.ones(int(rows.sum()))))
            y = target[rows]
            regulariser = self._penalty * np.eye(x.shape[1])
            regulariser[-1, -1] = 0.0
            self._weights[int(which)] = np.linalg.solve(x.T @ x + regulariser, x.T @ y)
            self._means[int(which)] = float(y.mean())
        return self

    @property
    def penalty(self) -> float:
        return self._penalty

    def predict(self, channel: Sequence[int], design: NDArray[np.float64]) -> NDArray[np.float64]:
        answers = np.full(len(channel), self._overall)
        for index, which in enumerate(channel):
            weights = self._weights.get(int(which))
            if weights is not None:
                answers[index] = float(np.append(design[index], 1.0) @ weights)
            elif int(which) in self._means:
                answers[index] = self._means[int(which)]
        return answers


@dataclass(frozen=True)
class ScoreRow:
    """The summed loss of one answer on one target, over the hidden cells of one kind."""

    corpus: str
    kind: str
    target: str
    answer: str
    cells: int
    loss: float
    trivial: float

    @property
    def r_squared(self) -> float:
        return 1.0 - self.loss / self.trivial if self.trivial > 0 else math.nan


def chosen_ridge(
    fitting: HiddenCells, stacking: HiddenCells, design: str, target: int
) -> PerChannelRidge:
    """The ridge map of ``design`` for ``target`` whose penalty errs least on the stacking cells."""
    truth = stacking.array("targets")[:, target]
    defined = _defined(stacking, target)
    best: PerChannelRidge | None = None
    best_error = math.inf
    for penalty in PENALTIES:
        ridge = PerChannelRidge(penalty).fit(
            fitting.channel,
            fitting.array(design),
            fitting.array("targets")[:, target],
            _defined(fitting, target),
        )
        guess = ridge.predict(stacking.channel, stacking.array(design))
        error = float(((guess[defined] - truth[defined]) ** 2).sum()) if defined.any() else 0.0
        if error < best_error:
            best, best_error = ridge, error
    if best is None:  # pragma: no cover - the grid is never empty
        raise ValueError("no penalty to choose from")
    return best


def answered(
    fitting: HiddenCells, stacking: HiddenCells, scoring: HiddenCells
) -> dict[str, NDArray[np.float64]]:
    """Every answer for the scoring cells, ``[cells, targets]``, the ridge maps fitted apart.

    A model's answer, where the scoring cells carry one, stands last and is fitted on nothing.
    """
    answers: dict[str, NDArray[np.float64]] = {
        "carry": scoring.array("carry"),
        "interpolation": scoring.array("interpolation"),
    }
    stacked_cross = np.zeros((len(stacking), len(TARGETS)))
    stacked_schedule = np.zeros((len(stacking), len(TARGETS)))
    cross = np.zeros((len(scoring), len(TARGETS)))
    schedule = np.zeros((len(scoring), len(TARGETS)))
    for index in range(len(TARGETS)):
        for design, store_stacked, store_scored in (
            ("cross", stacked_cross, cross),
            ("schedule", stacked_schedule, schedule),
        ):
            ridge = chosen_ridge(fitting, stacking, design, index)
            store_stacked[:, index] = _bounded(
                ridge.predict(stacking.channel, stacking.array(design)), index
            )
            store_scored[:, index] = _bounded(
                ridge.predict(scoring.channel, scoring.array(design)), index
            )
    answers["cross"], answers["schedule"] = cross, schedule
    stacked = np.zeros((len(scoring), len(TARGETS)))
    for index in range(len(TARGETS)):
        on_stack = np.column_stack(
            (
                stacking.array("carry")[:, index],
                stacking.array("interpolation")[:, index],
                stacked_cross[:, index],
                stacked_schedule[:, index],
            )
        )
        to_score = np.column_stack(
            (
                answers["carry"][:, index],
                answers["interpolation"][:, index],
                cross[:, index],
                schedule[:, index],
            )
        )
        stacked[:, index] = _bounded(
            PerChannelRidge()
            .fit(
                stacking.channel,
                on_stack,
                stacking.array("targets")[:, index],
                _defined(stacking, index),
            )
            .predict(scoring.channel, to_score),
            index,
        )
    answers["stacked"] = stacked
    if scoring.model:
        answers[MODEL] = scoring.array(MODEL)
    return answers


def _bounded(answers: NDArray[np.float64], target: int) -> NDArray[np.float64]:
    """A ridge map's answers, kept inside ``[RIDGE_EDGE, 1 - RIDGE_EDGE]`` for the presence."""
    bounded: NDArray[np.float64] = (
        np.clip(answers, RIDGE_EDGE, 1.0 - RIDGE_EDGE) if target == 0 else answers
    )
    return bounded


def _defined(cells: HiddenCells, target: int) -> NDArray[np.bool_]:
    """Which cells a target is defined on: every hidden cell for the presence, else the held."""
    present = cells.array("targets")[:, 0] > 0.5
    return np.ones(len(cells), dtype=bool) if target == 0 else present


def scored(
    corpus: str,
    scoring: HiddenCells,
    answers: Mapping[str, NDArray[np.float64]],
    *,
    presence_rate: float,
) -> list[ScoreRow]:
    """The losses summed per kind, target and answer, beside the trivial predictor's."""
    targets = scoring.array("targets")
    kinds = np.asarray(scoring.kind)
    rows = []
    for kind in KINDS:
        of_kind = np.ones(len(scoring), dtype=bool) if kind == "all" else kinds == kind
        if not of_kind.any():
            continue
        for index, target in enumerate(TARGETS):
            defined = of_kind & _defined(scoring, index)
            truth = targets[defined, index]
            if index == 0:
                trivial = _log_loss(np.full(len(truth), presence_rate), truth)
            else:
                trivial = float((truth**2).sum())
            for answer, given in answers.items():
                guess = given[defined, index]
                loss = (
                    _log_loss(guess, truth) if index == 0 else float(((guess - truth) ** 2).sum())
                )
                rows.append(
                    ScoreRow(corpus, kind, target, answer, int(defined.sum()), loss, trivial)
                )
    return rows


def _log_loss(probability: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    p = np.clip(probability, EDGE, 1.0 - EDGE)
    return float(-(truth * np.log(p) + (1.0 - truth) * np.log(1.0 - p)).sum())


@dataclass(frozen=True)
class CellsRow:
    """What a corpus's windows look like as cells: how many, how many hidden, how many held."""

    corpus: str
    windows: int
    rows_per_window: float
    cells_per_window: float
    hidden_cells: int
    held_share: float


def audit_corpus(
    corpus: str,
    training: Sequence[TokenWindow],
    validation: Sequence[TokenWindow],
    *,
    vocabulary_size: int,
    bins: int,
    strategy: MaskingStrategy,
    model: CellReconstruction | None = None,
    device: str = "cpu",
) -> tuple[list[ScoreRow], CellsRow]:
    """Every answer on every target of the corpus's hidden cells, scored on validation windows.

    A ``model`` answers the validation windows' hidden cells alone: it is fitted on nothing here.

    Raises:
        ValueError: If the fitting, the stacking or the scoring windows hold no hidden cell.
    """
    layout = BinnedTokens(bins)
    half = len(training) // 2
    fitting_grids = grids_of(training[:half], layout, strategy, seed=FIT_SEED)
    stacking_grids = grids_of(training[half:], layout, strategy, seed=FIT_SEED + 1)
    scoring_grids = grids_of(
        validation, layout, strategy, seed=VALIDATION_SEED, model=model, device=device
    )
    fitting = hidden_cells_of(fitting_grids, vocabulary_size=vocabulary_size, bins=bins)
    stacking = hidden_cells_of(stacking_grids, vocabulary_size=vocabulary_size, bins=bins)
    scoring = hidden_cells_of(scoring_grids, vocabulary_size=vocabulary_size, bins=bins)
    for name, cells in (("fitting", fitting), ("stacking", stacking), ("scoring", scoring)):
        if not len(cells):
            raise ValueError(
                f"{corpus}: the {name} windows hold no hidden cell; give the audit more windows"
            )
    presence_rate = float(
        np.mean([*fitting.array("targets")[:, 0], *stacking.array("targets")[:, 0]])
    )
    rows = scored(
        corpus, scoring, answered(fitting, stacking, scoring), presence_rate=presence_rate
    )
    counts = CellsRow(
        corpus=corpus,
        windows=len(scoring_grids),
        rows_per_window=float(np.mean([len(grid.rows) for grid in scoring_grids])),
        cells_per_window=float(np.mean([grid.hidden.size for grid in scoring_grids])),
        hidden_cells=len(scoring),
        held_share=float(np.mean(scoring.array("targets")[:, 0] > 0.5)) if len(scoring) else 0.0,
    )
    return rows, counts


def write_scores(path: Path, rows: Iterable[ScoreRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("corpus", "kind", "target", "answer", "cells", "loss", "trivial"))
        for row in rows:
            writer.writerow(
                (
                    row.corpus,
                    row.kind,
                    row.target,
                    row.answer,
                    row.cells,
                    f"{row.loss:.6g}",
                    f"{row.trivial:.6g}",
                )
            )


def read_scores(path: Path) -> list[ScoreRow]:
    with path.open(newline="") as handle:
        return [
            ScoreRow(
                r["corpus"],
                r["kind"],
                r["target"],
                r["answer"],
                int(r["cells"]),
                float(r["loss"]),
                float(r["trivial"]),
            )
            for r in csv.DictReader(handle)
        ]


def write_cells(path: Path, rows: Iterable[CellsRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "corpus",
                "windows",
                "rows_per_window",
                "cells_per_window",
                "hidden_cells",
                "held_share",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.corpus,
                    row.windows,
                    f"{row.rows_per_window:.2f}",
                    f"{row.cells_per_window:.2f}",
                    row.hidden_cells,
                    f"{row.held_share:.4f}",
                )
            )


def read_cells(path: Path) -> list[CellsRow]:
    with path.open(newline="") as handle:
        return [
            CellsRow(
                r["corpus"],
                int(r["windows"]),
                float(r["rows_per_window"]),
                float(r["cells_per_window"]),
                int(r["hidden_cells"]),
                float(r["held_share"]),
            )
            for r in csv.DictReader(handle)
        ]


def answers_table(rows: Sequence[ScoreRow], corpus: str, kind: str = "all") -> str:
    """R² of every answer on every target, over the hidden cells of one kind of one corpus."""
    chosen = [row for row in rows if row.corpus == corpus and row.kind == kind]
    lines = []
    for target in TARGETS:
        of_target = {row.answer: row for row in chosen if row.target == target}
        if not of_target:
            continue
        cells = next(iter(of_target.values())).cells
        lines.append(
            [target, str(cells)]
            + [
                _fmt(of_target[answer].r_squared) if answer in of_target else "—"
                for answer in _answers_in(chosen)
            ]
        )
    return table(["target", "cells", *_answers_in(chosen)], lines)


def kinds_table(rows: Sequence[ScoreRow], corpus: str, target: str) -> str:
    """R² of every answer on one target, kind of mask by kind of mask, for one corpus."""
    named = _answers_in([row for row in rows if row.corpus == corpus and row.target == target])
    lines = []
    for kind in KINDS:
        of_kind = {
            row.answer: row
            for row in rows
            if row.corpus == corpus and row.kind == kind and row.target == target
        }
        if not of_kind:
            continue
        cells = next(iter(of_kind.values())).cells
        lines.append(
            [kind, str(cells)]
            + [_fmt(of_kind[a].r_squared) if a in of_kind else "—" for a in named]
        )
    return table(["kind", "cells", *named], lines)


def _answers_in(rows: Iterable[ScoreRow]) -> list[str]:
    """The answers the rows hold, those without a model first, in the order they are named."""
    held = {row.answer for row in rows}
    return [answer for answer in (*ANSWERS, MODEL) if answer in held]


def cells_table(rows: Sequence[CellsRow]) -> str:
    return table(
        ["corpus", "windows", "rows / window", "cells / window", "hidden cells", "held"],
        [
            [
                r.corpus,
                str(r.windows),
                f"{r.rows_per_window:.1f}",
                f"{r.cells_per_window:.1f}",
                str(r.hidden_cells),
                f"{r.held_share:.2f}",
            ]
            for r in rows
        ],
    )


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def render(directory: Path) -> str:
    scores = read_scores(directory / SCORES)
    cells = read_cells(directory / CELLS)
    settings = (directory / SETTINGS).read_text()
    parts = [f"Settings:\n{settings}", "**Cells.**\n\n" + cells_table(cells)]
    for row in cells:
        parts.append(
            f"**{row.corpus}, every hidden cell.**\n\n" + answers_table(scores, row.corpus)
        )
        parts.append(
            f"**{row.corpus}, presence by kind of mask.**\n\n"
            + kinds_table(scores, row.corpus, "presence")
        )
        parts.append(
            f"**{row.corpus}, mean by kind of mask.**\n\n" + kinds_table(scores, row.corpus, "mean")
        )
    return "\n\n".join(parts)


def split_by_share(
    side: Side, training_units: Sequence[str], share: CorpusShare
) -> tuple[Side, Side]:
    """The training side's windows of the units the share reads, and of the units it leaves.

    The share chooses among the units the manifest lists, as the run chose, so a unit that
    yielded no window still counts towards the fraction.

    Raises:
        ValueError: If the share reads every unit, so none is left unread.
    """
    if share.is_whole:
        raise ValueError("the run read every training unit; none is left unread")
    read = set(share.select(training_units))
    sides = []
    for name, wanted in (("read", True), ("unread", False)):
        kept = [i for i, unit in enumerate(side.units) if (unit in read) is wanted]
        sides.append(
            Side(name, [side.windows[i] for i in kept], tuple(side.units[i] for i in kept))
        )
    return sides[0], sides[1]


def published_corpus(
    store: ArtifactStore,
    blocks: Path,
    manifest: ArtifactRef,
    corpus: str,
    *,
    training_windows: int,
    validation_windows: int,
) -> CorpusWindows:  # pragma: no cover - reads the store
    """A corpus no backbone has read, its windows as its publication cut them."""
    published = PublishedCorpusManifestJson().decode(store.get(manifest))
    return CorpusWindows.of(
        corpus,
        len(published.channels),
        published,
        fetched_block(store, published.block, blocks),
        training_windows=training_windows,
        validation_windows=validation_windows,
    )


def measure_published(
    arguments: argparse.Namespace, store: ArtifactStore
) -> None:  # pragma: no cover - the command line and the store
    """Score the answers without a model on one published corpus under the masks given."""
    stated = ExperimentFile.load(arguments.masks).configuration()
    if not stated.architecture.bins:
        raise SystemExit("no bins stated: the audit is over cells")
    key, checksum = arguments.manifest
    corpus = published_corpus(
        store,
        BLOCKS,
        ArtifactRef(key=key, checksum=Checksum.parse(checksum)),
        arguments.corpus,
        training_windows=arguments.training_windows,
        validation_windows=arguments.validation_windows,
    )
    rows, cells = audit_corpus(
        corpus.corpus,
        corpus.training,
        corpus.validation,
        vocabulary_size=corpus.vocabulary_size,
        bins=stated.architecture.bins,
        strategy=stated.masking,
        model=None,
        device="cpu",
    )
    out = arguments.out or WORKSPACE / f"{arguments.corpus}-{arguments.masks.stem}"
    out.mkdir(parents=True, exist_ok=True)
    write_scores(out / SCORES, rows)
    write_cells(out / CELLS, [cells])
    (out / SETTINGS).write_text(
        "\n".join(
            [
                f"measured: {datetime.now():%Y-%m-%d %H:%M}",
                f"commit: {SourceRevision().current()}",
                f"machine: {machine()}",
                f"manifest: {key} {checksum}",
                f"masks: {arguments.masks}",
                f"bins: {stated.architecture.bins}",
                f"strategy: {stated.masking}",
                f"training windows: {arguments.training_windows}",
                f"validation windows: {arguments.validation_windows}",
            ]
        )
        + "\n"
    )
    print(render(out))


def unread_corpus(
    backbone: Backbone,
    store: ArtifactStore,
    blocks: Path,
    corpus: str,
    *,
    training_windows: int,
    validation_windows: int,
) -> CorpusWindows:  # pragma: no cover - reads the registry's artifacts
    """The corpus's windows as the audit reads them where the backbone read it in part.

    The maps are fitted on windows of the units the run read and every answer is scored on
    windows of the units it left, both from the training side.

    Raises:
        SystemExit: If the backbone did not read the corpus, its block is not the one read, or
            it read every training unit.
    """
    stated = next((given for given in backbone.inputs if given.corpus == corpus), None)
    if stated is None:
        raise SystemExit(f"backbone {backbone.id} did not read {corpus}")
    manifest = PublishedCorpusManifestJson().decode(store.get(stated.manifest))
    if manifest.block.checksum != stated.block_checksum:
        raise SystemExit(f"{corpus}: the manifest's block is not the block the backbone read")
    training = sides_of(manifest, fetched_block(store, manifest.block, blocks))["training"]
    try:
        read, unread = split_by_share(
            training, manifest.training_units, backbone.configuration.corpus_share_of(corpus)
        )
    except ValueError as error:
        raise SystemExit(f"{corpus}: {error}") from error
    kept = evenly_spaced(len(unread.windows), validation_windows)
    return CorpusWindows(
        corpus=corpus,
        vocabulary_size=stated.vocabulary_size,
        channels_apart=channels_apart_of(manifest),
        training=[read.windows[i] for i in evenly_spaced(len(read.windows), training_windows)],
        validation=[unread.windows[i] for i in kept],
        validation_units=tuple(unread.units[i] for i in kept),
        block_checksum=str(manifest.block.checksum),
        channel_ids=frozenset(own_channels(manifest.channels, corpus)),
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backbone", help="registered backbone whose corpora are read as it read them"
    )
    parser.add_argument(
        "--manifest",
        nargs=2,
        metavar=("KEY", "CHECKSUM"),
        help="a published corpus no backbone has read, by its manifest; needs --corpus, --masks",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        help="experiment file whose [input] bins and [masking] are drawn; with --model, the "
        "backbone's own unless given",
    )
    parser.add_argument(
        "--model",
        choices=MODES,
        help="score the backbone's decoder as well: the weights kept, or as the run drew them",
    )
    parser.add_argument("--device", choices=DEVICES, default=available_device())
    parser.add_argument(
        "--corpus", help="one corpus to read: the backbone's own, or with --windows-of another's"
    )
    parser.add_argument(
        "--windows-of",
        help="the backbone whose windows of --corpus are read, where this one never read it",
    )
    parser.add_argument(
        "--unset-rows",
        choices=("drawn", "mean"),
        default="drawn",
        help="rows of the channels the backbone never read: drawn, or the mean of the learnt rows",
    )
    parser.add_argument(
        "--unread-units",
        action="store_true",
        help="score on the training units of --corpus the backbone's run left unread",
    )
    parser.add_argument("--out", type=Path, help="directory the CSV files are written to")
    parser.add_argument("--training-windows", type=int, default=TRAINING_WINDOWS)
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument(
        "--report-only", type=Path, help="render the tables of a directory already measured"
    )
    arguments = parser.parse_args(argv)
    if arguments.report_only is None and arguments.manifest is not None:
        if arguments.backbone is not None or arguments.model is not None:
            parser.error("--manifest reads a corpus no backbone read; it takes no --backbone")
        if arguments.corpus is None or arguments.masks is None:
            parser.error("--manifest needs --corpus, the corpus it holds, and --masks")
        return arguments
    if arguments.report_only is None:
        if arguments.backbone is None:
            parser.error("--backbone is needed to measure; --report-only to render")
        if arguments.masks is None and arguments.model is None:
            parser.error("--masks is needed unless --model reads the backbone's own")
        if arguments.windows_of is not None and arguments.corpus is None:
            parser.error("--windows-of needs --corpus, the corpus that backbone supplies")
        if arguments.windows_of is not None and arguments.model is None:
            parser.error("--windows-of reads a corpus the backbone never read; it needs --model")
        if arguments.unread_units and (arguments.corpus is None or arguments.windows_of):
            parser.error("--unread-units needs --corpus, one the backbone read itself")
    return arguments


def model_of(
    backbone: Backbone, mode: str, store: ArtifactStore
) -> CellReconstruction:  # pragma: no cover - reads the registry's artifacts
    """The backbone's objective with the weights kept, or as its run drew it before a step.

    Raises:
        SystemExit: If the backbone is not ready, continued another's weights where its
            initialisation is asked for, or was not pretrained over cells.
    """
    configuration = backbone.configuration
    if mode == "initialisation":
        if configuration.start is not None:
            raise SystemExit(
                f"backbone {backbone.id} continued another backbone's weights: its "
                "initialisation is that backbone's weights"
            )
        torch.manual_seed(configuration.budget.seed)
        model = ObjectiveShape.of(
            configuration, max(stated.vocabulary_size for stated in backbone.inputs)
        ).build()
    else:
        if backbone.artifact is None:
            raise SystemExit(f"backbone {backbone.id} is not ready: it has no weights")
        model = TrainedModel.read(store.get(backbone.artifact)).build()
    if not isinstance(model, CellReconstruction):
        raise SystemExit(f"backbone {backbone.id} was not pretrained over cells")
    return model


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    store = configured_store(Settings())
    if arguments.manifest is not None:
        measure_published(arguments, store)
        return
    registry = _registry()
    backbone = registry.get(_backbone_id(arguments.backbone, "--backbone"))
    if arguments.masks is not None:
        stated = ExperimentFile.load(arguments.masks).configuration()
        bins, strategy = stated.architecture.bins, stated.masking
    else:
        bins = backbone.configuration.architecture.bins
        strategy = backbone.configuration.masking
    if not bins:
        raise SystemExit("no bins stated: the audit is over cells")
    model = None
    if arguments.model is not None:
        if not device_available(arguments.device):
            raise SystemExit(f"device {arguments.device} is not available on this machine")
        if bins != backbone.configuration.architecture.bins:
            raise SystemExit(
                f"the masks lay {bins} bins, the backbone reads "
                f"{backbone.configuration.architecture.bins}"
            )
        model = model_of(backbone, arguments.model, store)
    source = backbone
    if arguments.windows_of is not None and model is not None:
        source = registry.get(_backbone_id(arguments.windows_of, "--windows-of"))
        try:
            supplied = supplied_by(source, backbone, arguments.corpus)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        # The rows no pretraining set are drawn from one seed, so a reading repeats.
        torch.manual_seed(VALIDATION_SEED)
        grown(model, max(given.vocabulary_size for given in supplied))
        if arguments.unset_rows == "mean":
            manifest = PublishedCorpusManifestJson().decode(store.get(supplied[0].manifest))
            unset_rows_to_mean(model, own_channels(manifest.channels, arguments.corpus))
    name = arguments.masks.stem if arguments.masks is not None else "own"
    if arguments.model is not None:
        name += f"-{arguments.model}"
    if arguments.windows_of is not None:
        name += f"-{arguments.corpus}-{arguments.unset_rows}"
    if arguments.unread_units:
        name += f"-{arguments.corpus}-unread"
    out = arguments.out or WORKSPACE / f"{arguments.backbone[:8]}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    if arguments.unread_units:
        corpora = [
            unread_corpus(
                backbone,
                store,
                BLOCKS,
                arguments.corpus,
                training_windows=arguments.training_windows,
                validation_windows=arguments.validation_windows,
            )
        ]
    else:
        corpora = corpora_of(
            source,
            store,
            BLOCKS,
            training_windows=arguments.training_windows,
            validation_windows=arguments.validation_windows,
            corpus=arguments.corpus,
        )
    scores: list[ScoreRow] = []
    counts: list[CellsRow] = []
    for corpus in corpora:
        started = time.perf_counter()
        rows, cells = audit_corpus(
            corpus.corpus,
            corpus.training,
            corpus.validation,
            vocabulary_size=corpus.vocabulary_size,
            bins=bins,
            strategy=strategy,
            model=model,
            device=arguments.device,
        )
        scores.extend(rows)
        counts.append(cells)
        seconds = time.perf_counter() - started
        print(
            f"{corpus.corpus}: {cells.hidden_cells} hidden cells in {seconds:.0f} s",
            file=sys.stderr,
        )
    write_scores(out / SCORES, scores)
    write_cells(out / CELLS, counts)
    (out / SETTINGS).write_text(
        "\n".join(
            [
                f"measured: {datetime.now():%Y-%m-%d %H:%M}",
                f"commit: {SourceRevision().current()}",
                f"machine: {machine()}",
                f"torch: {version('torch')}",
                f"backbone: {arguments.backbone}",
                f"masks: {arguments.masks or OWN_MASKS}",
                f"bins: {bins}",
                f"strategy: {strategy}",
                f"model: {arguments.model or 'none'}",
                f"device: {arguments.device if arguments.model else 'none'}",
                f"windows of: {arguments.windows_of or 'the backbone'}",
                f"unset rows: {arguments.unset_rows if arguments.windows_of else 'none'}",
                f"scored on: {SCORED_ON[arguments.unread_units]}",
                f"training windows: {arguments.training_windows}",
                f"validation windows: {arguments.validation_windows}",
                f"ridge penalties tried: {PENALTIES}, stack penalty: {STACK_PENALTY}",
                f"presence answers of the maps bounded at: {RIDGE_EDGE}",
                f"presence smoothing: {SMOOTHING}",
            ]
        )
        + "\n"
    )
    print(render(out))


if __name__ == "__main__":
    main()
