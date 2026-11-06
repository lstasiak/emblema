"""Whether a longer input lifts the first mission's detectors at the encoder's initialisation.

Every detector over the first mission reads one hour of telemetry, and the readouts taught every
event of the fitting side miss the same events: the hour shows nothing of them. Before a pretext
aims at this task, this report asks whether the input is the limit: the same detectors, over the
same encoder at its initialisation, read a window of several hours ending at the scored hour.
Nothing is pretrained and nothing is trained; the scored unit and its label stay the hour.

An arm is a layout ``<hours>x<bins>``: the window the encoder sees and the spans it is cut into.
Its cells are derived from the cells of the published one-hour windows, bin by bin, exactly as
the layout would cut the concatenated tokens, so no corpus is republished. Beside the layouts,
two families read context without changing what the encoder sees: ``pooled`` concatenates the
hour's state with the mean of the previous hours' states, and ``smoothed`` takes the maximum or
the mean of the one-hour detector's scores over the previous hours, the answer with no model at
all. The cells laid out as the encoder would read them stand as the classical bar at every
layout of six bins: the farthest single feature without a label, and trees taught the events.
Context never reads a frozen test month: a window after one starts its context at its own month.

Three steps, the choice between the arms made on fitting months that taught nothing:

- ``measure --section select``: the earlier fitting months fit every detector, the later ones
  are scored; the states of every layout and seed are kept so the pooled family reads them.
- ``choose``: for every family and detector, the arm with the highest mean area over the seeds
  on the selection months; of every arm within 0.01 of it the shortest, then the fewest bins.
- ``measure --section refit``: the chosen arms and the control, fitted on the whole fitting side
  and scored on the validation months; ``read`` sets each against the control there.

``measure --section lay`` lays the published cells out once, before the lanes start.
``measure`` touches torch and writes ``cells*.csv`` and ``answers/``, never an area; a lane
``i/n`` takes every n-th unit of work. A trial and a run never share a directory: the
conditions a run began under, ``--trial`` among them, are refused when they differ. ``check``
compares a cell's answers with a campaign's, printing distances and no area, so a trial stays
blind.

    uv run --env-file .env.r2 scripts/telemetry_context_report.py measure
        --task esa-mission1-events --task-id UUID --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --out DIR --section select
        --arms 1x6 6x6 24x6 72x6 6x36 24x24 --spans 6 24 72 --seeds 1 2 3
        --select-from 2004-01 [--lane 1/3] [--device mps] [--trial]
    uv run --env-file .env.r2 scripts/telemetry_context_report.py measure ... --section refit
    uv run scripts/telemetry_context_report.py choose --dir DIR [--partial]
    uv run scripts/telemetry_context_report.py read --dir DIR [--partial]
    uv run scripts/telemetry_context_report.py check --dir DIR --cell NAME --against ANSWERS.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from scripts.reporting import table
from scripts.taught_fine_tuning_pilot import (
    LEAST_GAIN,
    SHARE,
    Bench,
    bench_of,
    committed_tree,
    read_answers,
    record,
    refuse_another_commit,
    write_answers,
)
from scripts.telemetry_supervised_reference import TimeSides, month_of, time_sides

Floats = NDArray[np.float64]

LAY, SELECT, REFIT = "lay", "select", "refit"
SECTIONS = (LAY, SELECT, REFIT)
# What a window is read through: the encoder's state over a layout, the hour's state pooled
# with the previous hours', the hour's score smoothed over the previous hours', or the cells.
STATES, POOLED, SMOOTHED_MAX, SMOOTHED_MEAN, CELLS = (
    "states",
    "pooled",
    "smoothed-max",
    "smoothed-mean",
    "cells",
)
FAMILIES = (STATES, POOLED, SMOOTHED_MAX, SMOOTHED_MEAN, CELLS)
SMOOTHED = (SMOOTHED_MAX, SMOOTHED_MEAN)
# How a window is scored: the whitened distance without a label, Fisher's direction taught the
# events, the farthest single feature without a label, trees taught the events.
WHITENED, DIRECTION, FARTHEST, TREES = "whitened", "direction", "farthest", "trees"
DETECTORS_OVER_STATES = (WHITENED, DIRECTION)
DETECTORS_OVER_CELLS = (FARTHEST, TREES)
# The readings declared as the family under Holm; everything else is read beside them.
DECLARED = ((STATES, WHITENED), (STATES, DIRECTION))
# The spans of the published windows' layout, which every context layout is derived from.
BASE_BINS = 6
# Arms whose means on the selection months lie this close to the best are one choice.
WITHIN = 0.01
# About how many cells a batch holds across its windows, whatever the layout.
CELLS_PER_BATCH = 20_000
CELL_COLUMNS = (
    "cell",
    "section",
    "family",
    "arm",
    "detector",
    "seed",
    "events_taught",
    "fitting_windows",
    "scored_windows",
    "seconds",
)
# The published cells' feature axis, as the layout names it.
PRESENCE = BinnedTokens.PRESENCE
LOG_COUNT = BinnedTokens.LOG_COUNT
MEAN, MINIMUM, MAXIMUM, FIRST, LAST, OFFSET, SINCE = range(2, 9)


@dataclass(frozen=True, order=True)
class Layout:
    """A context layout: how many hours the encoder sees and how many spans they are cut into.

    Invariants: at least one hour and one bin; the published layout's spans over the hours
    divide into the bins, so every bin holds whole published cells.

    Attributes:
        hours: How many one-hour windows end at the scored hour and the hours before it.
        bins: Equal spans the hours are cut into.
    """

    hours: int
    bins: int

    def __post_init__(self) -> None:
        if self.hours < 1 or self.bins < 1:
            raise ValueError(f"a layout has hours and bins, got {self.name}")
        if (self.hours * BASE_BINS) % self.bins:
            raise ValueError(
                f"{self.name}: {self.hours * BASE_BINS} published cells do not cut into "
                f"{self.bins} bins"
            )

    @classmethod
    def parse(cls, text: str) -> "Layout":
        """The layout ``<hours>x<bins>`` names."""
        hours, sep, bins = text.partition("x")
        if not sep or not hours.isdigit() or not bins.isdigit():
            raise ValueError(f"a layout is <hours>x<bins>, not {text!r}")
        return cls(int(hours), int(bins))

    @property
    def name(self) -> str:
        return f"{self.hours}x{self.bins}"

    @property
    def cells_per_bin(self) -> int:
        """How many published cells of one channel each bin gathers."""
        return self.hours * BASE_BINS // self.bins


CONTROL = Layout(1, BASE_BINS)


def span_name(hours: int) -> str:
    """The arm name of a family that reads context in the detector, not the layout."""
    return f"{hours}h"


def hours_of(arm: str) -> int:
    """How many hours an arm's name spans, a layout's or a span's."""
    if arm.endswith("h"):
        return int(arm[:-1])
    return Layout.parse(arm).hours


def bins_of(arm: str) -> int:
    """How many bins an arm's name cuts, the control's for a span."""
    return BASE_BINS if arm.endswith("h") else Layout.parse(arm).bins


def context_cells(published: Floats, bins: int) -> Floats:
    """The cells of a context window from the published cells of the hours it spans.

    ``published`` is ``[..., cells, features]``: for each row, the published cells of the
    hours in time order, an hour that is not in the block as cells of nothing. The result is
    ``[..., bins, features]``, what the layout over ``bins`` would make of the concatenated
    tokens: every bin gathers whole published cells, so its count, level, extremes, ends and
    the offset of its last reading follow from theirs, and the carry forward and the "since"
    count run over the context's bins as they would over a window's.

    Raises:
        ValueError: If the published cells do not cut into ``bins``.
    """
    *lead, cells, features = published.shape
    if cells % bins:
        raise ValueError(f"{cells} published cells do not cut into {bins} bins")
    each = cells // bins
    grouped = published.reshape(*lead, bins, each, features)
    present = grouped[..., PRESENCE] > 0.0
    observed = present.any(axis=-1)
    counts = np.where(present, np.rint(np.expm1(grouped[..., LOG_COUNT])), 0.0)
    count = counts.sum(axis=-1)
    mean = np.where(
        observed, (counts * grouped[..., MEAN]).sum(axis=-1) / np.where(count > 0, count, 1.0), 0.0
    )
    minimum = np.where(observed, np.where(present, grouped[..., MINIMUM], np.inf).min(axis=-1), 0.0)
    maximum = np.where(
        observed, np.where(present, grouped[..., MAXIMUM], -np.inf).max(axis=-1), 0.0
    )
    earliest = present.argmax(axis=-1)[..., None]
    latest_cell = (each - 1 - present[..., ::-1].argmax(axis=-1))[..., None]
    first = np.take_along_axis(grouped[..., FIRST], earliest, axis=-1)[..., 0]
    last = np.take_along_axis(grouped[..., LAST], latest_cell, axis=-1)[..., 0]
    offset = (
        latest_cell[..., 0] + np.take_along_axis(grouped[..., OFFSET], latest_cell, axis=-1)[..., 0]
    ) / each
    offset = np.where(observed, np.minimum(offset, 1.0), 0.0)
    first, last = np.where(observed, first, 0.0), np.where(observed, last, 0.0)
    places = np.arange(bins)
    latest = np.maximum.accumulate(np.where(observed, places, -1), axis=-1)
    held = np.where(latest >= 0, np.take_along_axis(last, np.maximum(latest, 0), axis=-1), 0.0)
    since = np.where(latest >= 0, places - latest, places + 1) / bins
    for feature in (mean, minimum, maximum, first, last):
        feature[~observed] = held[~observed]
    stacked: Floats = np.stack(
        (
            observed.astype(np.float64),
            np.log1p(count),
            mean,
            minimum,
            maximum,
            first,
            last,
            offset,
            since,
        ),
        axis=-1,
    )
    return stacked


def laid_context(cells: Floats, rows: NDArray[np.int64], *, own_channels: bool) -> LaidWindow:
    """One context window's cells ``[rows, bins, features]`` as the layout lays a window out.

    Over the window's own channels, a row no bin observed is dropped, as the layout gives a
    window a row only for a channel it holds; over the corpus's, every row stays.
    """
    if own_channels:
        kept = (cells[..., PRESENCE] > 0.0).any(axis=-1)
        cells, rows = cells[kept], rows[kept]
    channels, bins, features = cells.shape
    return LaidWindow(
        features=cells.transpose(1, 0, 2).reshape(-1, features),
        channel_ids=np.tile(rows, bins),
        times=np.repeat((np.arange(bins) + 0.5) / bins, channels),
        timeless=np.zeros(channels * bins, dtype=bool),
    )


class PublishedCells:
    """The published one-hour cells of every loaded window, and the hours they end at.

    Every window of the fitting and validation months is laid out once over the corpus's
    channels; a context window gathers the hours before its own from here, so an hour the task
    never loaded, a frozen test month's among them, is an hour of nothing.
    """

    def __init__(self, cells: Floats, rows: NDArray[np.int64], ends_at: Sequence[float]) -> None:
        """``cells`` is ``[windows, rows, BASE_BINS, features]``, ``ends_at`` one hour per window.

        Raises:
            ValueError: If the windows do not end on whole hours, or two end at one.
        """
        hours = np.asarray(ends_at, dtype=np.float64)
        if not np.all(hours == np.rint(hours)):
            raise ValueError("the published windows end on whole hours")
        self._cells = cells
        self._rows = rows
        self._first = int(hours.min())
        self._at = np.full(int(hours.max()) - self._first + 1, -1, dtype=np.int64)
        places = np.rint(hours).astype(np.int64) - self._first
        if len(set(places.tolist())) != len(places):
            raise ValueError("two published windows end at one hour")
        self._at[places] = np.arange(len(hours))
        self._hours = np.rint(hours).astype(np.int64)

    @property
    def rows(self) -> NDArray[np.int64]:
        return self._rows

    @property
    def cells(self) -> Floats:
        """``[windows, rows, BASE_BINS, features]``, every loaded window's published cells."""
        return self._cells

    def context(self, indices: Sequence[int], layout: Layout) -> Floats:
        """``[len(indices), rows, bins, features]``: the windows' cells under ``layout``."""
        ends = self._hours[np.asarray(indices, dtype=np.int64)]
        spans = ends[:, None] - np.arange(layout.hours - 1, -1, -1)[None, :]
        places = spans - self._first
        inside = (places >= 0) & (places < len(self._at))
        found = np.where(inside, self._at[np.clip(places, 0, len(self._at) - 1)], -1)
        gathered = np.where(
            (found >= 0)[:, :, None, None, None],
            self._cells[np.maximum(found, 0)],
            0.0,
        )
        windows, hours, rows, bins, features = gathered.shape
        published = gathered.transpose(0, 2, 1, 3, 4).reshape(windows, rows, hours * bins, features)
        return context_cells(published, layout.bins)

    def previous(self, indices: Sequence[int], hours: int) -> NDArray[np.int64]:
        """``[len(indices), hours - 1]``: the loaded windows of the hours before each, or -1."""
        ends = self._hours[np.asarray(indices, dtype=np.int64)]
        spans = ends[:, None] - np.arange(hours - 1, 0, -1)[None, :]
        places = spans - self._first
        inside = (places >= 0) & (places < len(self._at))
        found: NDArray[np.int64] = np.where(
            inside, self._at[np.clip(places, 0, len(self._at) - 1)], -1
        )
        return found


def pooled_rows(
    states: Floats, cells: PublishedCells, indices: Sequence[int], hours: int
) -> Floats:
    """Each window's state beside the mean state of the loaded hours before it, or its own."""
    own = states[np.asarray(indices, dtype=np.int64)]
    before = cells.previous(indices, hours)
    present = (before >= 0)[:, :, None]
    gathered = np.where(present, states[np.maximum(before, 0)], 0.0)
    count = present.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beside = np.where(count > 0, gathered.sum(axis=1) / count, own)
    return np.concatenate((own, beside), axis=1)


def smoothed(
    predictions: Sequence[WindowPrediction], hours: int, *, how: str
) -> tuple[WindowPrediction, ...]:
    """Each scored window's score as the maximum or mean over the scored hours ending at it."""
    if how not in SMOOTHED:
        raise ValueError(f"a score is smoothed by one of {SMOOTHED}, not {how!r}")
    by_hour = {round(p.window.ends_at): p.predicted for p in predictions}
    reduce: Callable[[Sequence[float]], float] = max if how == SMOOTHED_MAX else fmean
    out = []
    for each in predictions:
        end = round(each.window.ends_at)
        scores = [by_hour[hour] for hour in range(end - hours + 1, end + 1) if hour in by_hour]
        out.append(
            WindowPrediction(window=each.window, target=each.target, predicted=reduce(scores))
        )
    return tuple(out)


def embedding_batch(layout: Layout, rows: int) -> int:
    """How many windows the encoder embeds at once: fewer the more cells a window has.

    The attention's memory grows with the square of the cells, so a batch of 64 windows of 288
    cells fits the accelerator and a batch of 64 windows of 1,728 cells spills out of it and
    runs thirty times slower than its arithmetic.
    """
    return max(4, min(64, CELLS_PER_BATCH // (rows * layout.bins)))


def cell_name(section: str, family: str, arm: str, detector: str, seed: int) -> str:
    return f"{section}-{family}-{arm}-{detector}-s{seed}"


TRUNCATED_COLUMNS = ("unit", "position", "ends_at", "target")


def month_of_hour(hour: int, origin_seconds: int) -> str:
    """The calendar month, ``YYYY-MM`` in UTC, of the hour ending ``hour`` hours past the origin."""
    instant = datetime.fromtimestamp(origin_seconds + (hour - 0.5) * 3600.0, tz=UTC)
    return instant.strftime("%Y-%m")


def truncated_windows(
    scored: Sequence[LabelledWindow],
    loaded_months: Collection[str],
    hours: int,
    origin_seconds: int,
) -> list[LabelledWindow]:
    """The scored windows whose context of ``hours`` reaches into a month the task never loaded.

    A month the task holds out, frozen, is never read, not even as context, so a window in the
    hours after one sees a context cut short. The selection months follow one another and lose
    nothing; the validation months interleave with the frozen ones, so a reading over every
    window would weigh the longer arms down there alone. One rule for every arm, the control
    included: these windows are not read.
    """
    found = []
    for window in scored:
        end = round(window.window.ends_at)
        before = range(end - hours + 1, end)
        if any(month_of_hour(hour, origin_seconds) not in loaded_months for hour in before):
            found.append(window)
    return found


def write_truncated(path: Path, windows: Iterable[LabelledWindow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRUNCATED_COLUMNS)
        for each in windows:
            writer.writerow(
                (each.window.unit, each.window.position, each.window.ends_at, each.target)
            )


def read_truncated(directory: Path, section: str) -> set[tuple[str, int]]:
    """Which windows of ``section`` are not read, by unit and place; none before ``measure``."""
    path = directory / f"truncated-{section}.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {(row["unit"], int(row["position"])) for row in csv.DictReader(handle)}


def kept(
    predictions: Iterable[WindowPrediction], dropped: Collection[tuple[str, int]]
) -> list[WindowPrediction]:
    """The predictions less the windows ``dropped``."""
    return [p for p in predictions if (str(p.window.unit), p.window.position) not in dropped]


def smoothed_cells(section: str, detector: str, seed: int, spans: Sequence[int]) -> list[str]:
    """The smoothed cells a control cell gives rise to: every family over every span."""
    return [
        cell_name(section, family, span_name(hours), detector, seed)
        for family in SMOOTHED
        for hours in spans
    ]


def chosen_of(directory: Path) -> list[dict[str, Any]]:
    """What ``choose`` wrote.

    Raises:
        SystemExit: If the choice has not been made.
    """
    path = directory / "chosen.json"
    if not path.exists():
        raise SystemExit(f"choose first: {path} is not there")
    loaded: list[dict[str, Any]] = json.loads(path.read_text())
    return loaded


@dataclass(frozen=True, kw_only=True)
class Work:
    """One unit of work of ``measure``: a family, an arm and a seed, with every cell it writes.

    Attributes:
        section: Which months fit and which are scored.
        family: What the windows are read through.
        arm: The layout or the span.
        seed: The seed, which draws the initialisation and the trees.
        detectors: The detectors the unit fits over its rows.
    """

    section: str
    family: str
    arm: str
    seed: int
    detectors: tuple[str, ...]

    def cells(self) -> list[str]:
        return [
            cell_name(self.section, self.family, self.arm, d, self.seed) for d in self.detectors
        ]


def cells_detectors(detectors: Sequence[str], seed: int, seeds: Sequence[int]) -> tuple[str, ...]:
    """The detectors over cells a seed fits: the farthest feature draws nothing, so once."""
    return tuple(d for d in detectors if d != FARTHEST or seed == min(seeds))


def select_work(
    layouts: Sequence[Layout], spans: Sequence[int], seeds: Sequence[int]
) -> list[Work]:
    """Every unit of the selection, the widest layouts first so the lanes end together."""
    work: list[Work] = []
    for layout in sorted(layouts, key=lambda each: (-each.hours * each.bins, each.name)):
        for seed in seeds:
            work.append(
                Work(
                    section=SELECT,
                    family=STATES,
                    arm=layout.name,
                    seed=seed,
                    detectors=DETECTORS_OVER_STATES,
                )
            )
        if layout.bins == BASE_BINS:
            for seed in seeds:
                work.append(
                    Work(
                        section=SELECT,
                        family=CELLS,
                        arm=layout.name,
                        seed=seed,
                        detectors=cells_detectors(DETECTORS_OVER_CELLS, seed, seeds),
                    )
                )
    for hours in spans:
        for seed in seeds:
            work.append(
                Work(
                    section=SELECT,
                    family=POOLED,
                    arm=span_name(hours),
                    seed=seed,
                    detectors=DETECTORS_OVER_STATES,
                )
            )
    return work


def lane_of(work: Sequence[Work], lane: str | None) -> list[Work]:
    """The units lane ``i/n`` takes, every n-th from the i-th; all of them without a lane.

    Raises:
        SystemExit: If the lane is not ``i/n`` with ``1 <= i <= n``.
    """
    if lane is None:
        return list(work)
    try:
        at, of = (int(part) for part in lane.split("/"))
    except ValueError:
        raise SystemExit(f"a lane is i/n, not {lane!r}") from None
    if not 1 <= at <= of:
        raise SystemExit(f"a lane is i/n with 1 <= i <= n, not {lane!r}")
    return [each for place, each in enumerate(work) if place % of == at - 1]


def lane_table(directory: Path, lane: str | None) -> Path:
    """Where a lane records its cells: one table per lane, so no two processes share a file."""
    if lane is None:
        return directory / "cells.csv"
    return directory / f"cells-{lane.replace('/', '-of-')}.csv"


def cells_of(directory: Path) -> list[dict[str, str]]:
    """Every cell every lane recorded, in the order recorded."""
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("cells*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def done_cells(directory: Path) -> set[str]:
    return {cell["cell"] for cell in cells_of(directory)}


def refuse_other_conditions(path: Path, conditions: Mapping[str, object]) -> None:
    """Refuse cells under other conditions than a run begun with.

    Raises:
        SystemExit: If ``path`` names other conditions.
    """
    if not path.exists():
        return
    begun = json.loads(path.read_text())
    differing = sorted(
        key
        for key in conditions
        if key != "commit" and key in begun and begun[key] != conditions[key]
    )
    if differing:
        raise SystemExit(f"this run began under other conditions: {', '.join(differing)}")


@dataclass(frozen=True, kw_only=True)
class Choice:
    """What one family and detector chose on the selection months.

    Attributes:
        family: What the windows are read through.
        detector: How they are scored.
        arm: The arm chosen: the shortest within the margin of the best.
        best_context: The arm of context with the highest mean, whether or not chosen.
        means: Each arm's mean area over the seeds.
    """

    family: str
    detector: str
    arm: str
    best_context: str | None
    means: Mapping[str, float]

    @property
    def chose_context(self) -> bool:
        return self.arm != CONTROL.name


def choose(family: str, detector: str, means: Mapping[str, float]) -> Choice:
    """The arm with the highest mean; of those within the margin the shortest, then the fewest bins.

    The control is among the arms, so a family whose context never beats it by the margin
    chooses the control.

    Raises:
        ValueError: If the control is not among the arms.
    """
    if CONTROL.name not in means:
        raise ValueError(f"{family} {detector}: the control is not among the arms")
    order = sorted(means, key=lambda arm: (hours_of(arm), bins_of(arm), arm))
    best = max(means.values())
    chosen = next(arm for arm in order if means[arm] >= best - WITHIN)
    context = [arm for arm in order if arm != CONTROL.name]
    best_context = max(context, key=lambda arm: (means[arm], -hours_of(arm))) if context else None
    return Choice(
        family=family, detector=detector, arm=chosen, best_context=best_context, means=means
    )


def mean_area(rankings: Mapping[int, AnomalyRanking]) -> float:
    """The mean area over the seeds."""
    return fmean(ranking.average_precision for ranking in rankings.values())


def mean_recall(rankings: Mapping[int, AnomalyRanking]) -> float:
    """The mean share of events found at the stated false alarms, over the seeds."""
    return fmean(r.event_recall_at_false_alarm_share(SHARE) for r in rankings.values())


def rankings_of(
    directory: Path, cells: Iterable[Mapping[str, str]], *, section: str
) -> dict[tuple[str, str], dict[str, dict[int, AnomalyRanking]]]:
    """Every cell's ranking of a section, by family and detector, then arm and seed."""
    found: dict[tuple[str, str], dict[str, dict[int, AnomalyRanking]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    dropped = read_truncated(directory, section)
    for cell in cells:
        if cell["section"] != section:
            continue
        answers = read_answers(directory / "answers" / f"{cell['cell']}.csv")
        found[(cell["family"], cell["detector"])][cell["arm"]][int(cell["seed"])] = (
            AnomalyRanking.of(kept(answers, dropped))
        )
    return found


def missing_selects(directory: Path, conditions: Mapping[str, Any]) -> list[str]:
    """The selection cells the conditions ask for that are not in."""
    work = select_work(
        [Layout.parse(arm) for arm in conditions["arms"]], conditions["spans"], conditions["seeds"]
    )
    held = done_cells(directory)
    wanted = [cell for each in work for cell in each.cells()]
    wanted += [
        cell
        for detector in DETECTORS_OVER_STATES
        for seed in conditions["seeds"]
        for cell in smoothed_cells(SELECT, detector, seed, conditions["spans"])
    ]
    return sorted(cell for cell in wanted if cell not in held)


def choose_from(directory: Path, *, partial: bool = False) -> list[Choice]:
    """Every family's choice from the selection months, written to ``chosen.json``.

    Raises:
        SystemExit: If a selection cell is missing and ``partial`` is not asked.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    if not partial:
        missing = missing_selects(directory, conditions)
        if missing:
            raise SystemExit(f"{len(missing)} selection cells are missing, the first {missing[0]}")
    found = rankings_of(directory, cells_of(directory), section=SELECT)
    choices = []
    for (family, detector), arms in sorted(found.items()):
        means = {arm: mean_area(seeds) for arm, seeds in arms.items()}
        if family != STATES:
            control = found.get((STATES if family != CELLS else CELLS, detector), {})
            if CONTROL.name in control and CONTROL.name not in means:
                means[CONTROL.name] = mean_area(control[CONTROL.name])
        if CONTROL.name not in means:
            if partial:
                continue
            raise SystemExit(f"{family} {detector}: no control among the selection cells")
        choices.append(choose(family, detector, means))
    (directory / "chosen.json").write_text(
        json.dumps(
            [
                {
                    "family": c.family,
                    "detector": c.detector,
                    "arm": c.arm,
                    "best_context": c.best_context,
                    "means": dict(sorted(c.means.items())),
                }
                for c in choices
            ],
            indent=2,
        )
        + "\n"
    )
    (directory / "chosen.md").write_text(choice_table(choices) + "\n")
    return choices


def choice_table(choices: Sequence[Choice]) -> str:
    arms = sorted(
        {arm for c in choices for arm in c.means}, key=lambda arm: (hours_of(arm), bins_of(arm))
    )
    return table(
        ("Family", "Detector", *arms, "Chosen", "Best context"),
        [
            (
                c.family,
                c.detector,
                *(f"{c.means[arm]:.3f}" if arm in c.means else "" for arm in arms),
                c.arm,
                c.best_context or "",
            )
            for c in choices
        ],
    )


def refit_work(directory: Path, seeds: Sequence[int]) -> list[Work]:
    """The units of the refit: each detector's chosen and best context arms, and the controls.

    An arm is fitted only under the detectors that chose it or read it beside; the control under
    every detector of its family.
    """
    detectors_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    for choice in chosen_of(directory):
        for arm in (choice["arm"], choice["best_context"]):
            if arm and arm != CONTROL.name:
                detectors_of[(choice["family"], arm)].add(choice["detector"])
    for detector in DETECTORS_OVER_STATES:
        detectors_of[(STATES, CONTROL.name)].add(detector)
    for detector in DETECTORS_OVER_CELLS:
        detectors_of[(CELLS, CONTROL.name)].add(detector)
    work = []
    for family in (STATES, CELLS, POOLED):
        named = sorted(
            (arm for found_family, arm in detectors_of if found_family == family),
            key=lambda arm: (-hours_of(arm) * bins_of(arm), arm),
        )
        order = DETECTORS_OVER_CELLS if family == CELLS else DETECTORS_OVER_STATES
        for arm in named:
            detectors = tuple(d for d in order if d in detectors_of[(family, arm)])
            for seed in seeds:
                fitted = cells_detectors(detectors, seed, seeds) if family == CELLS else detectors
                if fitted:
                    work.append(
                        Work(section=REFIT, family=family, arm=arm, seed=seed, detectors=fitted)
                    )
    return work


@dataclass(frozen=True, kw_only=True)
class ContextRow:
    """One family and detector: the arm read on the validation months against the control.

    Attributes:
        family: What the windows are read through.
        detector: How they are scored.
        arm: The arm read.
        chosen: Whether the arm is the one chosen, or the best context read beside.
        areas: The arm's area under each seed.
        control: The control's mean area.
        recall: The arm's share of events found at the stated false alarms, over the seeds.
        difference: The arm's mean less the control's.
        low: The interval's lower end, over months.
        high: The interval's upper end.
        p_value: The bootstrap's p-value.
        threshold: Twice the standard error over seeds; nothing here was pretrained.
        rejected: Whether the reading passes under the family's correction; ``None`` beside it,
            and for a row read alone, which the correction then does not touch.
    """

    family: str
    detector: str
    arm: str
    chosen: bool
    areas: Mapping[int, float]
    control: float
    recall: float
    difference: float
    low: float
    high: float
    p_value: float
    threshold: float
    rejected: bool | None = None

    @property
    def outcome(self) -> str:
        """What the reading says, by the declared table."""
        if not self.chosen:
            return "beside"
        if self.difference >= max(LEAST_GAIN, self.threshold) and self.low > 0.0:
            return "clearly above" if self.rejected is not False else "above, not under Holm"
        if self.difference < -self.threshold and self.high < 0.0:
            return "below"
        if self.threshold > LEAST_GAIN:
            return "undecided"
        return "around"


def context_row(
    family: str,
    detector: str,
    arm: str,
    candidate: Mapping[int, AnomalyRanking],
    control: Mapping[int, AnomalyRanking],
    *,
    chosen: bool,
    bootstrap: PairedUnitBootstrap,
) -> ContextRow:
    """The row of one arm against the control, seeds paired.

    Raises:
        SystemExit: If the control lacks a seed the arm holds.
    """
    seeds = sorted(candidate)
    if not set(seeds) <= set(control):
        raise SystemExit(f"{family} {detector} {arm}: the control lacks a seed the arm holds")
    differences = [
        candidate[seed].average_precision - control[seed].average_precision for seed in seeds
    ]
    compared = bootstrap.compare(
        PairedAnomalyRankings(
            control=tuple(control[seed] for seed in seeds),
            candidate=tuple(candidate[seed] for seed in seeds),
        )
    )
    standard_error = stdev(differences) / len(differences) ** 0.5 if len(differences) > 1 else 0.0
    return ContextRow(
        family=family,
        detector=detector,
        arm=arm,
        chosen=chosen,
        areas={seed: candidate[seed].average_precision for seed in seeds},
        control=fmean(control[seed].average_precision for seed in seeds),
        recall=fmean(candidate[seed].event_recall_at_false_alarm_share(SHARE) for seed in seeds),
        difference=fmean(differences),
        low=compared.interval.low,
        high=compared.interval.high,
        p_value=compared.p_value,
        threshold=DetectionThreshold(value=2.0 * standard_error).value,
    )


def corrected(rows: Sequence[ContextRow]) -> list[ContextRow]:
    """The declared rows judged as one family under Holm; the rest pass nothing."""
    family = [row for row in rows if (row.family, row.detector) in DECLARED and row.chosen]
    if not family:
        return list(rows)
    standing = HolmCorrection().rejected([row.p_value for row in family], family_size=len(DECLARED))
    judged = {id(row): rejected for row, rejected in zip(family, standing, strict=True)}
    return [replace(row, rejected=judged[id(row)]) if id(row) in judged else row for row in rows]


def context_table(rows: Sequence[ContextRow]) -> str:
    return table(
        (
            "Family",
            "Detector",
            "Arm",
            "Per seed",
            "Area",
            "Recall @1 %",
            "Control",
            "Difference [interval]",
            "p",
            "T",
            "Outcome",
            "Holm",
        ),
        [
            (
                row.family,
                row.detector,
                row.arm,
                " ".join(f"{area:.3f}" for area in row.areas.values()),
                f"{fmean(row.areas.values()):.3f}",
                f"{row.recall:.3f}",
                f"{row.control:.3f}",
                f"{row.difference:+.3f} [{row.low:+.3f}; {row.high:+.3f}]",
                f"{row.p_value:.3f}",
                f"{row.threshold:.3f}",
                row.outcome,
                "" if row.rejected is None else ("passes" if row.rejected else "does not pass"),
            )
            for row in rows
        ],
    )


def events_found(predictions: Sequence[WindowPrediction], share: float) -> frozenset[int]:
    """The events with a window ranked before the normal window that would exceed ``share``.

    What the ranking's recall at that share counts, as a set: the alarm is lowered from the top
    and stops before the normal window that takes the flagged share of normal windows past
    ``share``; an event is found if one of its windows scores above that window.
    """
    normal = sorted(
        (p.predicted for p in predictions if not AnomalyScheme.is_event(p.target)), reverse=True
    )
    if not normal:
        return frozenset()
    allowed = int(share * len(normal))
    boundary = normal[allowed] if allowed < len(normal) else -np.inf
    return frozenset(
        int(p.target)
        for p in predictions
        if AnomalyScheme.is_event(p.target) and p.predicted > boundary
    )


def found_table(
    directory: Path, cells: Iterable[Mapping[str, str]], *, section: str, share: float
) -> str:
    """Per cell, the events found at ``share`` of normal windows, and those the control misses."""
    by_cell = {cell["cell"]: cell for cell in cells if cell["section"] == section}
    dropped = read_truncated(directory, section)
    found = {
        name: events_found(
            kept(read_answers(directory / "answers" / f"{name}.csv"), dropped), share
        )
        for name in by_cell
    }
    rows = []
    for name, cell in sorted(by_cell.items()):
        control_family = CELLS if cell["family"] == CELLS else STATES
        control = cell_name(
            section, control_family, CONTROL.name, cell["detector"], int(cell["seed"])
        )
        against = found.get(control, frozenset())
        rows.append(
            (
                name,
                str(len(found[name])),
                str(len(found[name] - against)),
                str(len(against - found[name])),
                " ".join(str(e) for e in sorted(found[name] - against)),
            )
        )
    return table(("Cell", "Found", "Not by control", "Only by control", "Gained events"), rows)


def missing_refits(directory: Path, conditions: Mapping[str, Any]) -> list[str]:
    held = done_cells(directory)
    wanted = [cell for each in refit_work(directory, conditions["seeds"]) for cell in each.cells()]
    wanted += [
        cell
        for detector in DETECTORS_OVER_STATES
        for seed in conditions["seeds"]
        for cell in smoothed_cells(REFIT, detector, seed, conditions["spans"])
    ]
    return sorted(cell for cell in wanted if cell not in held)


def read_tables(directory: Path, *, partial: bool = False) -> str:
    """Every family's chosen arm against the control on the validation months.

    Raises:
        SystemExit: If a refit cell is missing and ``partial`` is not asked, or no cell is in.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    cells = cells_of(directory)
    if not cells:
        raise SystemExit(f"no cell in {directory}")
    if not partial:
        missing = missing_refits(directory, conditions)
        if missing:
            raise SystemExit(f"{len(missing)} refit cells are missing, the first {missing[0]}")
    chosen = {(c["family"], c["detector"]): c for c in chosen_of(directory)}
    found = rankings_of(directory, cells, section=REFIT)
    bootstrap = PairedUnitBootstrap()
    rows = []
    for (family, detector), arms in sorted(found.items()):
        control_family = CELLS if family == CELLS else STATES
        control = found.get((control_family, detector), {}).get(CONTROL.name)
        if control is None:
            if partial:
                continue
            raise SystemExit(f"{family} {detector}: no control among the refit cells")
        choice = chosen.get((family, detector))
        for arm, seeds in sorted(
            arms.items(), key=lambda each: (hours_of(each[0]), bins_of(each[0]))
        ):
            if arm == CONTROL.name:
                continue
            is_chosen = choice is not None and choice["arm"] == arm
            rows.append(
                context_row(
                    family, detector, arm, seeds, control, chosen=is_chosen, bootstrap=bootstrap
                )
            )
    rows = corrected(rows)
    declared = [row for row in rows if (row.family, row.detector) in DECLARED]
    beside = [row for row in rows if (row.family, row.detector) not in DECLARED]
    controls = table(
        ("Family", "Detector", "Control per seed", "Control", "Recall @1 %"),
        [
            (
                family,
                detector,
                " ".join(f"{control[s].average_precision:.3f}" for s in sorted(control)),
                f"{mean_area(control):.3f}",
                f"{mean_recall(control):.3f}",
            )
            for (family, detector), arms in sorted(found.items())
            if (control := arms.get(CONTROL.name))
        ],
    )
    dropped = read_truncated(directory, REFIT)
    not_read = (
        f"Windows not read, their context of {conditions.get('widest_hours', '?')} hours "
        f"reaching into a month the task never loaded: {len(dropped)} of "
        f"{conditions.get('windows_read', '?')}, under every arm alike."
    )
    text = "\n\n".join(
        (
            context_table(declared),
            context_table(beside),
            controls,
            not_read,
            found_table(directory, cells, section=REFIT, share=SHARE),
        )
    )
    (directory / "context.md").write_text(text + "\n")
    return text


def check_cell(directory: Path, cell: str, against: Path) -> str:
    """How far a cell's answers lie from another's over the same windows; no area is read.

    Raises:
        SystemExit: If the two score different windows.
    """
    own = {
        (str(p.window.unit), p.window.position): p
        for p in read_answers(directory / "answers" / f"{cell}.csv")
    }
    other = {(str(p.window.unit), p.window.position): p for p in read_answers(against)}
    if own.keys() != other.keys():
        raise SystemExit(
            f"{cell} scores {len(own)} windows, the other {len(other)}; "
            f"{len(own.keys() ^ other.keys())} differ"
        )
    keys = sorted(own)
    mine = np.array([own[k].predicted for k in keys])
    theirs = np.array([other[k].predicted for k in keys])
    targets_agree = all(own[k].target == other[k].target for k in keys)
    gap = np.abs(mine - theirs)
    rank_gap = np.abs(np.argsort(np.argsort(-mine)) - np.argsort(np.argsort(-theirs)))
    targets = "agree" if targets_agree else "DIFFER"
    return (
        f"{cell} against {against.name}: {len(keys)} windows, targets {targets}, "
        f"largest gap {gap.max():.2e}, mean {gap.mean():.2e}, within 1e-4: "
        f"{np.mean(gap < 1e-4):.4f}, ranks moved by more than 10: {int((rank_gap > 10).sum())}"
    )


# --- measuring, on the device ------------------------------------------------------------------


class Reader:  # pragma: no cover - environment
    """The task's loaded windows, their published cells, and the detectors fitted over them."""

    def __init__(self, bench: Bench, out: Path) -> None:
        from emblema.entrypoints.workers.known_detectors import KnownDetectors

        self.bench = bench
        self.out = out
        self.windows: tuple[LabelledWindow, ...] = (*bench.pool, *bench.read)
        self.at = {
            (str(w.window.unit), w.window.position): place for place, w in enumerate(self.windows)
        }
        self.catalogue = KnownDetectors.catalogue(next(iter(bench.weights.values())))
        self.shape = next(iter(bench.backbones.values()))
        if self.shape.bins != BASE_BINS:
            raise SystemExit(
                f"the shape reads {self.shape.bins} bins; the published cells have {BASE_BINS}"
            )
        self.cells = self._published_cells()
        self._cell_rows: tuple[tuple[str, tuple[int, ...]], Floats] | None = None

    def _published_cells(self) -> PublishedCells:
        """The one-hour cells of every loaded window over the corpus's channels, kept on disk."""
        manifest = self.bench.blocks.manifest_of(self.bench.task.manifest)
        observed = [c for c in manifest.channels if c.corpus == manifest.corpus and not c.timeless]
        statics = [c for c in manifest.channels if c.corpus == manifest.corpus and c.timeless]
        if statics:
            raise SystemExit("this report lays out measured channels; the corpus has static ones")
        layout = BinnedTokens(BASE_BINS, channels=[c.channel_id for c in observed])
        rows = np.asarray(layout.channels, dtype=np.int64)
        path = self.out / "cells-1h.npy"
        if not path.exists():
            started = time.perf_counter()
            block = self.bench.block
            laid = np.empty((len(self.windows), len(rows), BASE_BINS, BinnedTokens.N_FEATURES))
            for start in range(0, len(self.windows), 512):
                chunk = self.windows[start : start + 512]
                tokens = block.at([w.window.position for w in chunk])
                for place, each in enumerate(layout.lay(tokens)):
                    laid[start + place] = each.features.reshape(BASE_BINS, len(rows), -1).transpose(
                        1, 0, 2
                    )
            # Written whole under another name first, so a lane starting beside this one never
            # maps a file still being written.
            temporary = path.with_name(f"cells-1h.{os.getpid()}.tmp.npy")
            np.save(temporary, laid)
            os.replace(temporary, path)
            print(f"published cells laid in {time.perf_counter() - started:.0f} s", flush=True)
        cells = np.load(path, mmap_mode="r")
        return PublishedCells(cells, rows, [w.window.ends_at for w in self.windows])

    def sides(
        self, section: str, select_from: str
    ) -> tuple[list[int], list[int], list[LabelledWindow]]:
        """The fitting indices, the scored indices and the taught windows of ``section``."""
        if section == SELECT:
            cut: TimeSides = time_sides(self.bench.pool, select_from=select_from)
            fit, scored, taught = cut.teach, cut.read, list(cut.taught)
        else:
            fit, scored = self.bench.pool, self.bench.read
            taught = [w for w in fit if AnomalyScheme.is_event(w.target)]
        return (
            [self.at[(str(w.window.unit), w.window.position)] for w in fit],
            [self.at[(str(w.window.unit), w.window.position)] for w in scored],
            taught,
        )

    def states(self, layout: Layout, seed: int, device: str) -> Floats:
        """One state per loaded window under ``layout``, the encoder drawn by ``seed``; kept."""
        import torch

        from emblema.evaluation.adapters.torch.frozen_embedding import FrozenEmbedding
        from emblema.evaluation.contracts.identifiers import CandidateRef
        from emblema.shared.adapters.tensors.token_tensors import TokenTensors

        path = self.out / "states" / f"{layout.name}-s{seed}.npy"
        if path.exists():
            loaded: Floats = np.load(path)
            return loaded
        path.parent.mkdir(parents=True, exist_ok=True)
        recipe = self.catalogue.arm_of(
            CandidateRef("untrained_knn@distance=whitened,reference=unit")
        ).recipe(seed=seed)
        if recipe.reading is None:
            raise SystemExit("the states detector reads no encoder")
        window_layout = BinnedTokens(layout.bins)
        torch.manual_seed(recipe.seed)
        embedding = FrozenEmbedding.under(
            recipe.reading, self.shape, vocabulary_size=self.bench.channels, layout=window_layout
        ).to(device)
        embedding.eval()
        states = []
        batch_size = embedding_batch(layout, len(self.cells.rows))
        for start in range(0, len(self.windows), batch_size):
            indices = list(range(start, min(start + batch_size, len(self.windows))))
            cells = self.cells.context(indices, layout)
            laid = [
                laid_context(cells[i], self.cells.rows, own_channels=True)
                for i in range(len(indices))
            ]
            batch = TokenTensors.of(window_layout.batch(laid)).to(device)
            # Moved before it is widened: an accelerator without double precision hands back
            # zeros when asked to do both at once.
            states.append(embedding.embed(batch).to("cpu").to(torch.float64).numpy())
        stacked: Floats = np.concatenate(states)
        # Two lanes may need the control's states at once: each writes its own copy whole and
        # the last rename wins, so neither ever maps a file still being written.
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(temporary, stacked)
        os.replace(temporary, path)
        return stacked

    def cell_rows(self, layout: Layout, indices: Sequence[int]) -> Floats:
        """The cells over the corpus's channels, flattened, one row per window asked.

        The rows last laid are kept, so the seeds of one layout lay them once.
        """
        key = (layout.name, tuple(indices))
        if self._cell_rows is not None and self._cell_rows[0] == key:
            return self._cell_rows[1]
        rows = []
        for start in range(0, len(indices), 64):
            chunk = indices[start : start + 64]
            cells = self.cells.context(chunk, layout)
            rows.append(cells.transpose(0, 2, 1, 3).reshape(len(chunk), -1))
        laid: Floats = np.concatenate(rows)
        self._cell_rows = (key, laid)
        return laid

    def answers(
        self,
        detector: str,
        rows_fit: Floats,
        rows_scored: Floats,
        fit: Sequence[int],
        scored: Sequence[int],
        taught: Sequence[LabelledWindow],
        *,
        seed: int,
    ) -> tuple[WindowPrediction, ...]:
        """The detector fitted on the fitting rows and scored on the rest, against its month."""
        from emblema.evaluation.adapters.detection.feature_detectors import FeatureDetectors
        from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
        from emblema.evaluation.contracts.identifiers import CandidateRef

        base = "cells_knn" if detector in DETECTORS_OVER_CELLS else "untrained_knn"
        events = len({int(w.target) for w in taught})
        knobs = {
            WHITENED: "distance=whitened",
            FARTHEST: "distance=farthest",
            DIRECTION: f"events={events}",
            TREES: f"events={events},head=trees",
        }[detector]
        recipe = self.catalogue.arm_of(CandidateRef(f"{base}@{knobs},reference=unit")).recipe(
            seed=seed
        )
        fitting = [self.windows[i] for i in fit]
        taught_rows = (
            TaughtRows.among([w.window for w in fitting], taught)
            if detector in (DIRECTION, TREES)
            else None
        )
        fitted = FeatureDetectors.fitted(
            recipe, rows_fit, units=[str(w.window.unit) for w in fitting], taught=taught_rows
        )
        read = [self.windows[i] for i in scored]
        scores = fitted.scores(rows_scored, units=[str(w.window.unit) for w in read])
        return tuple(
            WindowPrediction(window=w.window, target=w.target, predicted=float(score))
            for w, score in zip(read, scores.tolist(), strict=True)
        )


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every unit of work of a section, on the device, those already answered skipped."""
    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = "trial" if arguments.trial else committed_tree()
    refuse_another_commit(out / "conditions.json", commit)
    layouts = [Layout.parse(arm) for arm in arguments.arms]
    if CONTROL not in layouts:
        raise SystemExit(f"the control {CONTROL.name} is among the arms")
    bench = bench_of(arguments)
    reader = Reader(bench, out)
    fit, scored, taught = reader.sides(arguments.section, arguments.select_from)
    widest = max(*(layout.hours for layout in layouts), *arguments.spans)
    truncated = truncated_windows(
        [reader.windows[i] for i in scored],
        {month_of(w) for w in reader.windows},
        widest,
        ORIGIN_SECONDS,
    )
    write_truncated(out / f"truncated-{arguments.section}.csv", truncated)
    conditions = {
        "commit": commit,
        "arms": [layout.name for layout in layouts],
        "spans": list(arguments.spans),
        "seeds": list(arguments.seeds),
        "select_from": arguments.select_from,
        "events_in_pool": len(
            {int(w.target) for w in bench.pool if AnomalyScheme.is_event(w.target)}
        ),
        "windows_loaded": len(reader.windows),
        "windows_read": len(bench.read),
        "months_read": len({month_of(w) for w in bench.read}),
        "widest_hours": widest,
        f"windows_not_read_in_{arguments.section}": len(truncated),
        f"events_touched_in_{arguments.section}": len(
            {int(w.target) for w in truncated if AnomalyScheme.is_event(w.target)}
        ),
        "device": arguments.device,
        "trial": bool(arguments.trial),
    }
    refuse_other_conditions(out / "conditions.json", conditions)
    if not (out / "conditions.json").exists():
        (out / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
    elif arguments.lane is None or arguments.lane.startswith("1/"):
        # The first lane of a section adds the section's own counts to the conditions.
        begun = json.loads((out / "conditions.json").read_text())
        missing = {key: value for key, value in conditions.items() if key not in begun}
        if missing:
            (out / "conditions.json").write_text(json.dumps({**begun, **missing}, indent=2) + "\n")
    if arguments.section == LAY:
        return
    if arguments.section == SELECT:
        work = select_work(layouts, arguments.spans, arguments.seeds)
    else:
        work = refit_work(out, arguments.seeds)
    table_path = lane_table(out, arguments.lane)
    if not table_path.exists():
        with table_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(CELL_COLUMNS)
    done = done_cells(out)
    events_taught = len({int(w.target) for w in taught})
    for unit in lane_of(work, arguments.lane):
        if all(cell in done for cell in unit.cells()):
            continue
        started = time.perf_counter()
        if unit.family == STATES:
            states = reader.states(Layout.parse(unit.arm), unit.seed, arguments.device)
            rows_fit, rows_scored = states[fit], states[scored]
        elif unit.family == POOLED:
            states = reader.states(CONTROL, unit.seed, arguments.device)
            rows_fit = pooled_rows(states, reader.cells, fit, hours_of(unit.arm))
            rows_scored = pooled_rows(states, reader.cells, scored, hours_of(unit.arm))
        else:
            layout = Layout.parse(unit.arm)
            rows_fit, rows_scored = reader.cell_rows(layout, fit), reader.cell_rows(layout, scored)
        laid_in = time.perf_counter() - started
        for detector in unit.detectors:
            cell = cell_name(unit.section, unit.family, unit.arm, detector, unit.seed)
            if cell in done:
                continue
            begun = time.perf_counter()
            answers = reader.answers(
                detector, rows_fit, rows_scored, fit, scored, taught, seed=unit.seed
            )
            write_answers(out / "answers" / f"{cell}.csv", answers)
            seconds = time.perf_counter() - begun + laid_in
            laid_in = 0.0
            row = (unit.section, unit.family, unit.arm, detector, unit.seed)
            record(table_path, (cell, *row, events_taught, len(fit), len(scored), f"{seconds:.1f}"))
            print(f"{cell}: {seconds:.0f} s", flush=True)
        if unit.family != STATES or unit.arm != CONTROL.name:
            continue
        # The answer with no model: the control's own scores over the previous hours. Derived
        # from the control's answers on disk, so a cut run picks up the ones it lost.
        for detector in unit.detectors:
            names = smoothed_cells(unit.section, detector, unit.seed, arguments.spans)
            if all(name in done for name in names):
                continue
            cell = cell_name(unit.section, unit.family, unit.arm, detector, unit.seed)
            answers = tuple(read_answers(out / "answers" / f"{cell}.csv"))
            for family in SMOOTHED:
                for hours in arguments.spans:
                    name = cell_name(unit.section, family, span_name(hours), detector, unit.seed)
                    if name in done:
                        continue
                    write_answers(
                        out / "answers" / f"{name}.csv", smoothed(answers, hours, how=family)
                    )
                    row = (unit.section, family, span_name(hours), detector, unit.seed)
                    record(table_path, (name, *row, events_taught, len(fit), len(scored), "0.0"))


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="fit and score every cell of a section")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True)
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument(
        "--weights", nargs=3, action="append", required=True, metavar=("NAME", "KEY", "CHECKSUM")
    )
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--section", choices=SECTIONS, required=True)
    measuring.add_argument(
        "--arms", nargs="+", default=["1x6", "6x6", "24x6", "72x6", "6x36", "24x24"]
    )
    measuring.add_argument("--spans", nargs="+", type=int, default=[6, 24, 72])
    measuring.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    measuring.add_argument("--select-from", required=True, help="the first month that chooses")
    measuring.add_argument("--lane", default=None, help="i/n: this process takes every n-th unit")
    measuring.add_argument("--device", default="mps")
    measuring.add_argument("--trial", action="store_true")
    choosing = commands.add_parser("choose", help="the choice from the selection months")
    choosing.add_argument("--dir", type=Path, required=True)
    choosing.add_argument("--partial", action="store_true")
    reading = commands.add_parser("read", help="the reading on the validation months")
    reading.add_argument("--dir", type=Path, required=True)
    reading.add_argument("--partial", action="store_true")
    checking = commands.add_parser("check", help="a cell's answers against another's")
    checking.add_argument("--dir", type=Path, required=True)
    checking.add_argument("--cell", required=True)
    checking.add_argument("--against", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.command == "measure":
        measure(arguments)
    elif arguments.command == "choose":
        print(choice_table(choose_from(arguments.dir, partial=arguments.partial)))
    elif arguments.command == "read":
        print(read_tables(arguments.dir, partial=arguments.partial))
    else:
        print(check_cell(arguments.dir, arguments.cell, arguments.against))


if __name__ == "__main__":
    main()
