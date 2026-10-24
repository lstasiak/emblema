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
    uv run scripts/cell_pretext_audit.py --report-only data/report/cell-pretext/<run>
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

from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.training.cell_batches import CellBatches
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.kernel.tokens import TokenWindow
from scripts.corpus_saturation_report import BLOCKS
from scripts.pretext_triviality_report import (
    BATCH_SIZE,
    FIT_SEED,
    TRAINING_WINDOWS,
    VALIDATION_SEED,
    VALIDATION_WINDOWS,
    CorpusWindows,
    _backbone_id,
    _registry,
    corpora_of,
)
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "cell-pretext"
TARGETS = ("presence", "log_count", "mean", "minimum", "maximum")
ANSWERS = ("carry", "interpolation", "cross", "schedule", "stacked")
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
    which of them are hidden, and the bin. ``carry`` and ``interpolation`` are answers already.
    """

    channel: list[int] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    targets: list[NDArray[np.float64]] = field(default_factory=list)
    carry: list[NDArray[np.float64]] = field(default_factory=list)
    interpolation: list[NDArray[np.float64]] = field(default_factory=list)
    cross: list[NDArray[np.float64]] = field(default_factory=list)
    schedule: list[NDArray[np.float64]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.channel)

    def array(self, name: str) -> NDArray[np.float64]:
        rows = getattr(self, name)
        return np.stack(rows) if rows else np.zeros((0, len(TARGETS)))


@dataclass(frozen=True)
class Grid:
    """One window as the grids an audit reads: full, visible, hidden and the kind of each cell."""

    rows: NDArray[np.int64]
    full: NDArray[np.float64]
    visible: NDArray[np.float64]
    hidden: NDArray[np.bool_]
    kind: NDArray[np.str_]


def grids_of(
    windows: Sequence[TokenWindow],
    layout: BinnedTokens,
    strategy: MaskingStrategy,
    *,
    seed: int,
) -> list[Grid]:
    """Each window as its grids, under the masks the strategy draws from ``seed``, batch by batch.

    The timeless cells are left aside: a static feature has no span to be hidden over.
    """
    batches = CellBatches(layout)
    grids = []
    for start in range(0, len(windows), BATCH_SIZE):
        chunk = windows[start : start + BATCH_SIZE]
        collated = batches.collate(chunk)
        masks = TokenMasking(strategy).draw(collated, torch.Generator().manual_seed(seed + start))
        scored = batches.scored(collated, masks)
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
    """Every answer for the scoring cells, ``[cells, targets]``, the ridge maps fitted apart."""
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
            for answer in ANSWERS:
                guess = answers[answer][defined, index]
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
) -> tuple[list[ScoreRow], CellsRow]:
    """Every answer on every target of the corpus's hidden cells, scored on validation windows.

    Raises:
        ValueError: If the fitting, the stacking or the scoring windows hold no hidden cell.
    """
    layout = BinnedTokens(bins)
    half = len(training) // 2
    fitting_grids = grids_of(training[:half], layout, strategy, seed=FIT_SEED)
    stacking_grids = grids_of(training[half:], layout, strategy, seed=FIT_SEED + 1)
    scoring_grids = grids_of(validation, layout, strategy, seed=VALIDATION_SEED)
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
                for answer in ANSWERS
            ]
        )
    return table(["target", "cells", *ANSWERS], lines)


def kinds_table(rows: Sequence[ScoreRow], corpus: str, target: str) -> str:
    """R² of every answer on one target, kind of mask by kind of mask, for one corpus."""
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
            + [_fmt(of_kind[a].r_squared) if a in of_kind else "—" for a in ANSWERS]
        )
    return table(["kind", "cells", *ANSWERS], lines)


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


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backbone", help="registered backbone whose corpora are read as it read them"
    )
    parser.add_argument(
        "--masks", type=Path, help="experiment file whose [input] bins and [masking] are drawn"
    )
    parser.add_argument("--corpus", help="one corpus of the backbone's inputs, instead of all")
    parser.add_argument("--out", type=Path, help="directory the CSV files are written to")
    parser.add_argument("--training-windows", type=int, default=TRAINING_WINDOWS)
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument(
        "--report-only", type=Path, help="render the tables of a directory already measured"
    )
    arguments = parser.parse_args(argv)
    if arguments.report_only is None and (arguments.backbone is None or arguments.masks is None):
        parser.error("--backbone and --masks are needed to measure; --report-only to render")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the command line
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    stated = ExperimentFile.load(arguments.masks).configuration()
    if not stated.architecture.reads_cells:
        raise SystemExit(f"{arguments.masks} states no [input] bins; the audit is over cells")
    backbone = _registry().get(_backbone_id(arguments.backbone, "--backbone"))
    store = configured_store(Settings())
    out = arguments.out or WORKSPACE / f"{arguments.backbone[:8]}-{arguments.masks.stem}"
    out.mkdir(parents=True, exist_ok=True)
    corpora: list[CorpusWindows] = corpora_of(
        backbone,
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
            bins=stated.architecture.bins,
            strategy=stated.masking,
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
                f"masks: {arguments.masks}",
                f"bins: {stated.architecture.bins}",
                f"strategy: {stated.masking}",
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
