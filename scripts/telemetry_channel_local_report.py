"""Read the first mission through each channel's own recent hours, with no model first.

The events the one-hour readouts miss are local, touch few channels and are short. A detector
over the mean of every channel's cells sees little of such an event, and a longer window
averaged the same way sees less. This report reads each channel's hour against that channel's
own previous hours: the departure of the hour's summary from the median of the hours before it,
over their spread, floored; a window's score is the largest departure over its channels, or the
mean, with no model and no label. Beside it, the same departures taught every event: Fisher's
direction over them, over them beside the encoder's state of the hour, and trees. The controls
are the direction and the whitened distance over the untrained encoder's states, and the cells'
farthest feature, refitted here on the same windows.

Four commands, so a reading stays blind until the choice is made:

- ``measure``: the departures of every loaded window under each span and feature set, kept
  on disk, and every cell's answers of a section (``select`` chooses on the later fitting
  months, ``refit`` reads the validation months); no area is written.
- ``choose``: for each family and detector, the arm with the highest mean area on the
  selection months; within 0.01 of it the shorter span, then the hour's summary before the
  bins.
- ``choose-recall``: of the no-model arms, the one finding the most events at 1 % of normal
  windows on a choosing section above those it finds with every month's scores turned, the
  measure the first mission's verdict is read by less its chance; beside, the events found
  averaged over 0.5 to 2 % and by their first six hours, and the short events' excess.
  ``--trial`` turns every month's scores first, so the whole path runs at chance before the
  choice is declared. The sections that
  choose are the selection months and the months before them (``early``), which ``measure``
  scores by the no-model arms alone, the floor and the alarm set on the selection months.
  ``--leave-out`` names event classes whose windows are not scored, so a class that one period
  holds and the validation months hardly do does not decide the choice.
- ``read``: the chosen arms on the validation months against their controls, the two declared
  readings under Holm; which events each cell finds, their locality, and how many of its top
  false alarms follow an event.

The cells and the states come from the context report's run, laid and embedded once.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS
from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.reporting import table
from scripts.taught_fine_tuning_pilot import (
    SHARE,
    bench_of,
    committed_tree,
    read_answers,
    record,
    refuse_another_commit,
    write_answers,
)
from scripts.telemetry_context_report import (
    CONTROL,
    DIRECTION,
    FARTHEST,
    REFIT,
    SELECT,
    TREES,
    WHITENED,
    ContextRow,
    PublishedCells,
    Reader,
    check_cell,
    context_table,
    events_found,
    kept,
    read_truncated,
    refuse_other_conditions,
    truncated_windows,
    write_truncated,
)
from scripts.telemetry_supervised_reference import month_of

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]

# The months before the selection months, read by the no-model arms alone with the floor and the
# alarm set on the selection months: a second side to choose on, which no arm has been read on.
EARLY = "early"
SECTIONS = (SELECT, REFIT, EARLY)
CHOOSING = (SELECT, EARLY)
# What the windows are read through: each channel's departure from its own recent hours with no
# model, the departures taught the events, or the controls over the states and the cells.
LOCAL, TAUGHT, STATES, CELLS = "local", "taught", "states", "cells"
# The detectors of the departures: the largest over the channels, the mean over the channels
# present, Fisher's direction over them, the direction over them beside the hour's state, trees.
MAX, MEAN, COMBINED = "max", "mean", "combined"
# Added beside before the reading, after the selection months read the direction over the 59
# departures at chance while the largest departure read 0.17: the events lie in which channel
# departs most, an order statistic no weight per channel expresses, so these read the direction
# over the largest few departures, alone and beside the hour's state.
TOP, ORDERED = "top", "ordered"
LARGEST = 3
LOCAL_DETECTORS = (MAX, MEAN)
TAUGHT_DETECTORS = (DIRECTION, COMBINED, TREES, TOP, ORDERED)
SEEDED = (COMBINED, TREES, ORDERED)
# The two readings declared as one family under Holm: whether the departures beside the state
# read above the state, and whether the largest departure reads above the cells' bar.
DECLARED = ((TAUGHT, COMBINED), (LOCAL, MAX))
# What an hour of a channel is summarised as: its bins' means averaged, the least of their
# minima, the largest of their maxima, the last reading, and the log of the readings counted.
HOUR, BINS = "hour", "bins"
FEATURE_SETS = (HOUR, BINS)
SUMMARY_NAMES = ("mean", "min", "max", "last", "log_count")
_FEATURES = BinnedTokens.FEATURE_NAMES
_MEAN, _MIN, _MAX, _LAST = (
    _FEATURES.index(name) for name in ("mean", "minimum", "maximum", "last")
)
_PRESENCE, _LOG_COUNT = BinnedTokens.PRESENCE, BinnedTokens.LOG_COUNT
# A channel's local spread is floored at this share of its spread over the fitting side, so a
# quantised channel whose recent hours happen to agree cannot read a step of one count as an
# event; the same share floors the standardisation of the package's detectors.
FLOOR_SHARE = 0.1
DEVIATION_PER_MAD = 1.4826
# A seedless cell is recorded under this seed, and read against every seed of a seeded control.
NO_SEED = 0
# The validation events that none of the supervised reference's seven arms nor any arm of the
# context report found at 1 % of normal windows under two seeds (telemetry-context.md, checks
# after the run); read descriptively, since the local few-channel profile was read off them.
UNFOUND_BEFORE = frozenset({40, 50, 110, 132, 136, 148, 156, 157, 166, 183, 184})
RESAMPLES = 10_000
CELL_COLUMNS = (
    "cell",
    "section",
    "family",
    "detector",
    "arm",
    "seed",
    "events_taught",
    "fitting_windows",
    "scored_windows",
    "seconds",
)


@dataclass(frozen=True)
class Arm:
    """A feature set and a span of hours the hour is read against."""

    features: str
    hours: int

    @classmethod
    def parse(cls, text: str) -> Arm:
        """``<features>-<hours>h``, as the arms are named.

        Raises:
            ValueError: If the text is not of that form.
        """
        features, _, span = text.partition("-")
        if features not in FEATURE_SETS or not span.endswith("h") or not span[:-1].isdigit():
            raise ValueError(f"an arm is <features>-<hours>h, not {text!r}")
        hours = int(span[:-1])
        if hours < 2:
            raise ValueError(f"{text}: a span reads at least one hour before the scored one")
        return cls(features, hours)

    @property
    def name(self) -> str:
        return f"{self.features}-{self.hours}h"

    @property
    def least_context(self) -> int:
        """How many of the previous hours must be loaded for the hour to be read: half."""
        return -(-(self.hours - 1) // 2)


def arms_of(features: Sequence[str], spans: Sequence[int]) -> list[Arm]:
    return [Arm(each, hours) for each in features for hours in spans]


def hour_summaries(cells: Floats) -> Floats:
    """``[windows, rows, 5]``: each channel's hour summarised, missing where no bin holds it."""
    present = cells[..., _PRESENCE] > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        count = present.sum(axis=-1)
        mean = np.where(present, cells[..., _MEAN], 0.0).sum(axis=-1) / count
        least = np.where(present, cells[..., _MIN], np.inf).min(axis=-1)
        most = np.where(present, cells[..., _MAX], -np.inf).max(axis=-1)
        last_bin = cells.shape[-2] - 1 - np.argmax(present[..., ::-1], axis=-1)
        last = np.take_along_axis(cells[..., _LAST], last_bin[..., None], axis=-1)[..., 0]
        readings = np.where(present, np.expm1(cells[..., _LOG_COUNT]), 0.0).sum(axis=-1)
    summaries = np.stack((mean, least, most, last, np.log1p(readings)), axis=-1)
    return np.where((count > 0)[..., None], summaries, np.nan)


def bin_values(cells: Floats) -> Floats:
    """``[windows, rows, bins, 5]``: each bin's values, missing where the bin holds nothing."""
    present = cells[..., _PRESENCE] > 0.0
    values = np.stack(
        (
            cells[..., _MEAN],
            cells[..., _MIN],
            cells[..., _MAX],
            cells[..., _LAST],
            cells[..., _LOG_COUNT],
        ),
        axis=-1,
    )
    return np.where(present[..., None], values, np.nan)


def nanmedian(values: Floats, axis: int, *, keepdims: bool = False) -> Floats:
    """The median along ``axis`` over the values present; missing where none is, without a word."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median: Floats = np.nanmedian(values, axis=axis, keepdims=keepdims)
    return median


def spread_of(values: Floats, axis: int) -> Floats:
    """The scaled median absolute deviation along ``axis``, missing values left out."""
    centre = nanmedian(values, axis=axis, keepdims=True)
    return DEVIATION_PER_MAD * nanmedian(np.abs(values - centre), axis=axis)


def global_floor(summaries: Floats) -> Floats:
    """``[rows, 5]``: the least local spread, a share of the channel's spread on the fitting side.

    A channel without spread on the fitting side takes the share of the median spread over the
    channels and summaries that have one; where none has, one.
    """
    with np.errstate(invalid="ignore"):
        spread = np.nan_to_num(spread_of(summaries, axis=0), nan=0.0)
    above = spread[spread > 0.0]
    typical = float(np.median(above)) if len(above) else 1.0 / FLOOR_SHARE
    return FLOOR_SHARE * np.where(spread > 0.0, spread, typical)


def context_statistics(
    context: Floats, *, per_hour: int = 1
) -> tuple[Floats, Floats, NDArray[np.int64]]:
    """The previous hours' median, spread and count per channel and value.

    ``context`` is ``[windows, m, rows, 5]``, missing where an hour or a channel is absent,
    ``per_hour`` values of each hour side by side along the second axis; the median and the
    scaled median absolute deviation are over the values present, and the count is of the
    hours in which the channel holds anything.
    """
    centre = nanmedian(context, axis=1)
    spread = spread_of(context, axis=1)
    windows, samples, rows, values = context.shape
    by_hour = (~np.isnan(context)).reshape(windows, samples // per_hour, per_hour, rows, values)
    loaded = by_hour.any(axis=(2, 4)).sum(axis=1)
    return centre, spread, loaded


def departures_from(
    now: Floats,
    centre: Floats,
    spread: Floats,
    loaded: NDArray[np.int64],
    floor: Floats,
    *,
    least_context: int,
) -> Floats:
    """``[windows, rows]``: each channel's largest departure of the hour from its recent hours.

    ``now`` is ``[windows, rows, k, 5]``, the hour's values (one summary, or its bins). The
    departure is the distance from the context's median over its spread, floored at ``floor``;
    the largest over the values and the summaries. A channel absent from the hour, or with fewer
    than ``least_context`` previous hours loaded, departs nought.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.maximum(np.nan_to_num(spread, nan=0.0), floor[None])
        departures = np.abs(now - centre[:, :, None, :]) / scale[:, :, None, :]
        largest = np.nanmax(np.where(np.isnan(departures), -np.inf, departures), axis=(2, 3))
    enough = loaded >= least_context
    return np.where(enough & np.isfinite(largest), largest, 0.0)


def local_deviations(now: Floats, context: Floats, floor: Floats, *, least_context: int) -> Floats:
    """The departures of ``now`` from ``context``, the statistics and the floor in one step."""
    centre, spread, loaded = context_statistics(context)
    return departures_from(now, centre, spread, loaded, floor, least_context=least_context)


def largest_departures(deviations: Floats, count: int = LARGEST) -> Floats:
    """``[windows, count]``: each window's largest departures in order, logged, the rest left."""
    ordered = np.sort(np.log1p(deviations), axis=1)[:, ::-1]
    return np.ascontiguousarray(ordered[:, :count])


def window_scores(deviations: Floats, present: NDArray[np.bool_], how: str) -> Floats:
    """Each window's score: the largest departure over its channels, or their mean."""
    if how == MAX:
        largest: Floats = deviations.max(axis=1)
        return largest
    if how != MEAN:
        raise ValueError(
            f"a window's score is the {MAX} or the {MEAN} of its channels, not {how!r}"
        )
    count = present.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: Floats = np.where(
            count > 0, np.where(present, deviations, 0.0).sum(axis=1) / count, 0.0
        )
    return mean


class Departures:
    """The departures of every loaded window under an arm, computed in chunks and kept on disk."""

    CHUNK = 1024

    def __init__(self, cells: PublishedCells, directory: Path) -> None:
        self.cells = cells
        self.directory = directory
        self._summaries: Floats | None = None

    def summaries(self) -> Floats:
        """``[windows, rows, 5]`` over every loaded window; kept beside the departures."""
        if self._summaries is not None:
            return self._summaries
        path = self.directory / "summaries.npy"
        if path.exists():
            self._summaries = np.load(path)
            return self._summaries
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.cells.cells
        chunks = [
            hour_summaries(np.asarray(raw[start : start + self.CHUNK]))
            for start in range(0, len(raw), self.CHUNK)
        ]
        self._summaries = np.concatenate(chunks)
        np.save(path, self._summaries)
        return self._summaries

    def present(self) -> NDArray[np.bool_]:
        """``[windows, rows]``: whether the channel holds a reading in the window's hour."""
        return ~np.isnan(self.summaries()[..., 0])

    def floor(self, fit: Sequence[int]) -> Floats:
        return global_floor(self.summaries()[np.asarray(fit, dtype=np.int64)])

    def statistics(self, arm: Arm) -> tuple[Floats, Floats, NDArray[np.int64]]:
        """The context's median, spread and count under ``arm`` for every window; kept.

        The statistics do not depend on the section, so both sections share them and only the
        floor, from each section's fitting side, is applied anew.
        """
        paths = [
            self.directory / f"{arm.name}-{name}.npy" for name in ("centre", "spread", "loaded")
        ]
        if all(path.exists() for path in paths):
            centre, spread, loaded = (np.load(path) for path in paths)
            return centre, spread, loaded
        summaries = self.summaries()
        raw = self.cells.cells
        parts: list[tuple[Floats, Floats, NDArray[np.int64]]] = []
        # The bins' context gathers six cells an hour, so its chunks are a sixth as long.
        chunk = self.CHUNK if arm.features == HOUR else max(1, self.CHUNK // 6)
        for start in range(0, len(summaries), chunk):
            indices = list(range(start, min(start + chunk, len(summaries))))
            before = self.cells.previous(indices, arm.hours)
            found = before >= 0
            if arm.features == HOUR:
                context = np.where(
                    found[:, :, None, None], summaries[np.maximum(before, 0)], np.nan
                )
            else:
                gathered = np.where(
                    found[:, :, None, None, None],
                    bin_values(np.asarray(raw[np.maximum(before, 0).ravel()])).reshape(
                        (*before.shape, *raw.shape[1:3], len(SUMMARY_NAMES))
                    ),
                    np.nan,
                )
                windows, hours, rows, bins, values = gathered.shape
                context = gathered.transpose(0, 1, 3, 2, 4).reshape(
                    windows, hours * bins, rows, values
                )
            parts.append(context_statistics(context, per_hour=context.shape[1] // (arm.hours - 1)))
        centre = np.concatenate([part[0] for part in parts])
        spread = np.concatenate([part[1] for part in parts])
        loaded = np.concatenate([part[2] for part in parts])
        self.directory.mkdir(parents=True, exist_ok=True)
        for path, array in zip(paths, (centre, spread, loaded), strict=True):
            np.save(path, array)
        return centre, spread, loaded

    def now(self, arm: Arm, indices: Sequence[int]) -> Floats:
        """``[len(indices), rows, k, 5]``: the hour's values the arm reads, a summary or bins."""
        if arm.features == HOUR:
            return self.summaries()[np.asarray(indices, dtype=np.int64)][:, :, None, :]
        return bin_values(np.asarray(self.cells.cells[np.asarray(indices, dtype=np.int64)]))

    def of(self, arm: Arm, section: str, fit: Sequence[int]) -> Floats:
        """``[windows, rows]`` under ``arm``, the floor from the section's fitting side; kept."""
        path = self.directory / f"{section}-{arm.name}.npy"
        if path.exists():
            kept_departures: Floats = np.load(path)
            return kept_departures
        floor = self.floor(fit)
        centre, spread, loaded = self.statistics(arm)
        out = []
        for start in range(0, len(centre), self.CHUNK):
            indices = list(range(start, min(start + self.CHUNK, len(centre))))
            out.append(
                departures_from(
                    self.now(arm, indices),
                    centre[indices],
                    spread[indices],
                    loaded[indices],
                    floor,
                    least_context=arm.least_context,
                )
            )
        departures: Floats = np.concatenate(out)
        np.save(path, departures)
        return departures


@dataclass(frozen=True)
class Choice:
    """The arm chosen for a family and detector on the selection months.

    Attributes:
        family: What the windows are read through.
        detector: How they are scored.
        arm: The arm chosen.
        means: Each arm's mean area over the seeds.
    """

    family: str
    detector: str
    arm: str
    means: Mapping[str, float]


def choose(family: str, detector: str, means: Mapping[str, float]) -> Choice:
    """The arm with the highest mean area; within 0.01 of it the shorter span, then the hour.

    Raises:
        ValueError: If no arm was measured.
    """
    if not means:
        raise ValueError(f"{family} {detector}: no arm to choose from")
    best = max(means.values())
    near = [Arm.parse(arm) for arm, mean in means.items() if mean >= best - 0.01]
    chosen = min(near, key=lambda arm: (arm.hours, FEATURE_SETS.index(arm.features)))
    return Choice(family=family, detector=detector, arm=chosen.name, means=dict(means))


def cells_of(directory: Path) -> list[dict[str, str]]:
    path = directory / "cells.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def done_cells(directory: Path) -> set[str]:
    return {row["cell"] for row in cells_of(directory)}


def cell_name(section: str, family: str, detector: str, arm: str | None, seed: int) -> str:
    name = f"{section}-{family}-{detector}"
    if arm is not None:
        name += f"-{arm}"
    if seed != NO_SEED:
        name += f"-s{seed}"
    return name


def arm_cells(
    section: str, arms: Sequence[Arm], seeds: Sequence[int]
) -> list[tuple[str, str, str, int]]:
    """Every (family, detector, arm, seed) of the arms, seedless where the detector draws none."""
    cells = []
    for arm in arms:
        for detector in LOCAL_DETECTORS:
            cells.append((LOCAL, detector, arm.name, NO_SEED))
        for detector in TAUGHT_DETECTORS:
            for seed in seeds if detector in SEEDED else (NO_SEED,):
                cells.append((TAUGHT, detector, arm.name, seed))
    return cells


def control_cells(seeds: Sequence[int]) -> list[tuple[str, str, int]]:
    """Every (family, detector, seed) of the controls."""
    cells = [(STATES, detector, seed) for detector in (DIRECTION, WHITENED) for seed in seeds]
    cells.append((CELLS, FARTHEST, NO_SEED))
    return cells


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
    arms = arms_of(conditions["features"], conditions["spans"])
    wanted = [
        cell_name(SELECT, family, detector, arm, seed)
        for family, detector, arm, seed in arm_cells(SELECT, arms, conditions["seeds"])
    ]
    held = done_cells(directory)
    return sorted(cell for cell in wanted if cell not in held)


def choose_from(directory: Path, *, partial: bool = False) -> list[Choice]:
    """The choices from the selection cells on disk.

    Raises:
        SystemExit: If selection cells are missing and the choice is not marked partial.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    missing = missing_selects(directory, conditions)
    if missing and not partial:
        raise SystemExit(f"{len(missing)} selection cells are missing, the first {missing[0]}")
    rankings = rankings_of(directory, cells_of(directory), section=SELECT)
    choices = []
    for (family, detector), arms in sorted(rankings.items()):
        if family in (STATES, CELLS):
            continue
        means = {
            arm: fmean(r.average_precision for r in seeds.values()) for arm, seeds in arms.items()
        }
        choices.append(choose(family, detector, means))
    return choices


def choice_table(choices: Sequence[Choice]) -> str:
    arms = sorted(
        {arm for choice in choices for arm in choice.means},
        key=lambda arm: (Arm.parse(arm).hours, FEATURE_SETS.index(Arm.parse(arm).features)),
    )
    return table(
        ("Family", "Detector", *arms, "Chosen"),
        [
            (
                choice.family,
                choice.detector,
                *(f"{choice.means[arm]:.3f}" if arm in choice.means else "" for arm in arms),
                choice.arm,
            )
            for choice in choices
        ],
    )


# The choice by the events found, the measure the first mission's verdict is read by. An event
# lasts eight hours at the median and up to fifteen days, so a score drawn at random per window
# finds about a third of the selection months' events at 1 % of false alarms, and how many a
# detector finds by chance depends on how its alarms are spread: the arms are chosen by the
# events found above those found when every month's scores are turned, the same turn for every
# arm. Beside: the most found averaged over four shares of normal windows, so a choice that rests
# on one operating point shows; and the events found by their first six hours (the standard span),
# where a long event has no more hours than a typical one to cross the alarm by chance.
AVERAGED_SHARES = (0.005, 0.01, 0.015, 0.02)
ONSET_HOURS = 6
# An event of at most this many windows is short. Where a side's events run longer than on the
# months the bar is read on, its short events' excess is read beside the choice: there length
# gives chance little to work with.
SHORT_WINDOWS = 6
ROTATIONS = 1000
ROTATION_SEED = 1
RECALL_COLUMNS = (
    "detector",
    "arm",
    "events",
    "found",
    "chance",
    "excess",
    "averaged",
    "onset",
    "short_events",
    "short_found",
    "short_chance",
    "short_excess",
    "chosen",
)


def month_slices(units: Ints) -> list[tuple[int, int]]:
    """Where each unit's run of windows starts and how long it is.

    Raises:
        ValueError: If a unit's windows are not contiguous.
    """
    changes = np.flatnonzero(np.diff(units)) + 1
    starts = np.concatenate(([0], changes))
    lengths = np.diff(np.concatenate((starts, [len(units)])))
    if len(set(units[starts].tolist())) != len(starts):
        raise ValueError("a unit's windows are not contiguous")
    return list(zip(starts.tolist(), lengths.tolist(), strict=True))


def rotation_order(slices: Sequence[tuple[int, int]], draws: np.random.Generator) -> Ints:
    """An index that turns every month by a random offset of at least one window."""
    parts = []
    for start, length in slices:
        offset = int(draws.integers(1, length)) if length > 1 else 0
        parts.append(start + (np.arange(length) + offset) % length)
    return np.concatenate(parts)


def alarm_boundary(scores: Floats, events: Ints, share: float) -> float:
    """The score a window must pass to be flagged at ``share`` of the normal windows.

    The alarm stops before the normal window that would flag more than ``share`` of them, so a
    window that ties with that normal window is not flagged, as the harness counts it.
    """
    normal = scores[events == 0]
    allowed = int(share * len(normal))
    if allowed >= len(normal):
        return -np.inf
    return float(np.partition(normal, len(normal) - 1 - allowed)[len(normal) - 1 - allowed])


def found_mask(scores: Floats, events: Ints, event_ids: Ints, share: float) -> Bools:
    """Which of ``event_ids`` hold a window above the alarm at ``share`` of normal windows."""
    flagged = np.unique(events[(scores > alarm_boundary(scores, events, share)) & (events > 0)])
    return np.isin(event_ids, flagged)


@dataclass(frozen=True)
class SideArrays:
    """One arm's answers on a side as arrays, the windows by unit and then by place.

    Attributes:
        units: Each window's unit, numbered in the units' order.
        positions: Where the block holds each window, so runs of alarms can be told apart.
        ends: Each window's end, in hours.
        events: Each window's event, nought for none.
        scores: Each window's score.
    """

    units: Ints
    positions: Ints
    ends: Floats
    events: Ints
    scores: Floats

    @classmethod
    def of(cls, predictions: Sequence[WindowPrediction]) -> SideArrays:
        rows = sorted(predictions, key=lambda p: (str(p.window.unit), p.window.position))
        names = sorted({str(p.window.unit) for p in rows})
        number = {name: place for place, name in enumerate(names)}
        return cls(
            units=np.asarray([number[str(p.window.unit)] for p in rows], dtype=np.int64),
            positions=np.asarray([p.window.position for p in rows], dtype=np.int64),
            ends=np.asarray([p.window.ends_at for p in rows], dtype=np.float64),
            events=np.asarray([int(p.target) for p in rows], dtype=np.int64),
            scores=np.asarray([p.predicted for p in rows], dtype=np.float64),
        )

    def turned(self, order: Ints) -> SideArrays:
        """The same windows with the scores taken in ``order``."""
        return replace(self, scores=self.scores[order])

    @property
    def event_ids(self) -> Ints:
        return np.unique(self.events[self.events > 0])

    @property
    def short(self) -> Bools:
        """Whether each of ``event_ids`` holds at most ``SHORT_WINDOWS`` windows."""
        _, counts = np.unique(self.events[self.events > 0], return_counts=True)
        return np.asarray(counts <= SHORT_WINDOWS)


def onset_windows(arrays: SideArrays, hours: int) -> Bools:
    """Whether each window ends within ``hours`` of its event's first scored window's end.

    The onset is the first window the side scores, which for an event begun before the side's
    first month is not the event's own.
    """
    first = {
        int(event): float(arrays.ends[arrays.events == event].min()) for event in arrays.event_ids
    }
    starts = np.asarray([first.get(int(event), np.inf) for event in arrays.events])
    return (arrays.events > 0) & (arrays.ends < starts + hours)


def found_count(arrays: SideArrays, share: float, among: Bools | None = None) -> int:
    """How many events the alarm at ``share`` finds, by the windows ``among`` names if given."""
    flagged = arrays.scores > alarm_boundary(arrays.scores, arrays.events, share)
    flagged &= arrays.events > 0
    if among is not None:
        flagged &= among
    return len(np.unique(arrays.events[flagged]))


def chance_found(
    arms: Mapping[tuple[str, str], SideArrays], *, rotations: int, seed: int
) -> dict[tuple[str, str], tuple[float, float]]:
    """Each arm's events found on average when every month's scores are turned, one turn for all.

    Beside the events found in all, the short events found.

    Raises:
        ValueError: If the arms do not hold the same windows and events.
    """
    first = next(iter(arms.values()))
    for key, arrays in arms.items():
        same = np.array_equal(arrays.units, first.units) and np.array_equal(
            arrays.events, first.events
        )
        if not same:
            raise ValueError(f"{key[0]} {key[1]} holds other windows or events than the first")
    slices = month_slices(first.units)
    draws = np.random.default_rng(seed)
    totals = dict.fromkeys(arms, 0)
    short_totals = dict.fromkeys(arms, 0)
    for _ in range(rotations):
        order = rotation_order(slices, draws)
        for key, arrays in arms.items():
            found = found_mask(
                arrays.scores[order],
                arrays.events,
                arrays.event_ids,
                AnomalyRanking.PRODUCT_FALSE_ALARM_SHARE,
            )
            totals[key] += int(found.sum())
            short_totals[key] += int(found[arrays.short].sum())
    return {key: (totals[key] / rotations, short_totals[key] / rotations) for key in arms}


@dataclass(frozen=True)
class RecallReading:
    """An arm's events found on a choosing section.

    Attributes:
        detector: How the channels' departures become the window's score.
        arm: The feature set and span.
        events: The events the side scores.
        found: The events found at the product's share of false alarms.
        chance: The events found on average with every month's scores turned.
        averaged: The events found, averaged over ``AVERAGED_SHARES``.
        onset: The events found by their first ``ONSET_HOURS`` at the product's share.
        short_events: The events of at most ``SHORT_WINDOWS`` windows.
        short_found: The short events found at the product's share.
        short_chance: The short events found on average with every month's scores turned.
    """

    detector: str
    arm: str
    events: int
    found: int
    chance: float
    averaged: float
    onset: int
    short_events: int
    short_found: int
    short_chance: float

    @property
    def excess(self) -> float:
        return self.found - self.chance

    @property
    def short_excess(self) -> float:
        return self.short_found - self.short_chance


def recall_reading(
    detector: str, arm: str, arrays: SideArrays, chance: tuple[float, float]
) -> RecallReading:
    share = AnomalyRanking.PRODUCT_FALSE_ALARM_SHARE
    found = found_mask(arrays.scores, arrays.events, arrays.event_ids, share)
    return RecallReading(
        detector=detector,
        arm=arm,
        events=len(arrays.event_ids),
        found=int(found.sum()),
        chance=chance[0],
        averaged=fmean(found_count(arrays, each) for each in AVERAGED_SHARES),
        onset=found_count(arrays, share, onset_windows(arrays, ONSET_HOURS)),
        short_events=int(arrays.short.sum()),
        short_found=int(found[arrays.short].sum()),
        short_chance=chance[1],
    )


def choose_by_recall(readings: Sequence[RecallReading]) -> RecallReading:
    """The reading with the largest excess over chance; then the most found and the averaged.

    Among readings alike in all three the plainest arm stands: the shorter span, the hour before
    the bins, the largest channel before the mean.

    Raises:
        ValueError: If there is no reading.
    """
    if not readings:
        raise ValueError("no arm to choose from")

    def order(reading: RecallReading) -> tuple[float, int, float, int, int, int]:
        arm = Arm.parse(reading.arm)
        return (
            round(reading.excess, 9),
            reading.found,
            round(reading.averaged, 9),
            -arm.hours,
            -FEATURE_SETS.index(arm.features),
            -LOCAL_DETECTORS.index(reading.detector),
        )

    return max(readings, key=order)


def selection_answers(
    directory: Path, section: str = SELECT
) -> dict[tuple[str, str], list[WindowPrediction]]:
    """Every no-model arm's answers on a choosing ``section``, by detector and arm.

    Raises:
        SystemExit: If an arm's cell is missing, or two arms score other windows or other events.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    dropped = read_truncated(directory, section)
    answers: dict[tuple[str, str], list[WindowPrediction]] = {}
    for arm in arms_of(conditions["features"], conditions["spans"]):
        for detector in LOCAL_DETECTORS:
            name = cell_name(section, LOCAL, detector, arm.name, NO_SEED)
            path = directory / "answers" / f"{name}.csv"
            if not path.exists():
                raise SystemExit(f"{name} is missing: every arm is read before one is chosen")
            answers[(detector, arm.name)] = kept(read_answers(path), dropped)
    keys = {
        key: sorted((str(p.window.unit), p.window.position, p.target) for p in rows)
        for key, rows in answers.items()
    }
    first = next(iter(keys.values()))
    for key, held in keys.items():
        if held != first:
            raise SystemExit(f"{key[0]} {key[1]} scores other windows or events than the first")
    return answers


@dataclass(frozen=True)
class LeftOut:
    """Events not scored by a choice, and how to tell the windows they touch.

    A window's target names only the lowest event it overlaps, so the windows are told by every
    event they overlap.

    Attributes:
        classes: The event numbers of each class left out.
        overlaps: Every scored event a window overlaps.
    """

    classes: Mapping[str, Sequence[int]]
    overlaps: Callable[[TaskWindow], Sequence[int]]

    @property
    def events(self) -> frozenset[int]:
        return frozenset(number for each in self.classes.values() for number in each)

    @property
    def stem(self) -> str:
        return "".join(f"-without-{kind}" for kind in sorted(self.classes))


def without_events(
    predictions: Sequence[WindowPrediction], left_out: LeftOut | None
) -> list[WindowPrediction]:
    """The predictions less every window touching an event left out: neither event nor normal."""
    if left_out is None:
        return list(predictions)
    events = left_out.events
    return [
        p
        for p in predictions
        if int(p.target) not in events and not events.intersection(left_out.overlaps(p.window))
    ]


def recall_readings(
    directory: Path,
    *,
    section: str = SELECT,
    rotations: int = ROTATIONS,
    trial: int | None = None,
    left_out: LeftOut | None = None,
) -> list[RecallReading]:
    """Each no-model arm's events found on a choosing ``section``, with its chance beside.

    The windows of the events ``left_out`` are not scored, so they neither find nor raise alarms.

    With ``trial``, every month's scores are first turned by an offset drawn from that seed, the
    same for every arm, so the whole reading runs at chance before the choice is declared.
    """
    answers = selection_answers(directory, section)
    arms = {key: SideArrays.of(without_events(rows, left_out)) for key, rows in answers.items()}
    if trial is not None:
        first = next(iter(arms.values()))
        order = rotation_order(month_slices(first.units), np.random.default_rng(trial))
        arms = {key: arrays.turned(order) for key, arrays in arms.items()}
    chance = chance_found(arms, rotations=rotations, seed=ROTATION_SEED)
    return [
        recall_reading(detector, arm, arrays, chance[(detector, arm)])
        for (detector, arm), arrays in arms.items()
    ]


def recall_rows(readings: Sequence[RecallReading]) -> list[dict[str, object]]:
    chosen = choose_by_recall(readings)
    return [
        {
            "detector": r.detector,
            "arm": r.arm,
            "events": r.events,
            "found": r.found,
            "chance": f"{r.chance:.3f}",
            "excess": f"{r.excess:.3f}",
            "averaged": f"{r.averaged:.2f}",
            "onset": r.onset,
            "short_events": r.short_events,
            "short_found": r.short_found,
            "short_chance": f"{r.short_chance:.3f}",
            "short_excess": f"{r.short_excess:.3f}",
            "chosen": int(r == chosen),
        }
        for r in readings
    ]


def recall_table(rows: Sequence[Mapping[str, str]]) -> str:
    """The choice's table from its CSV: events of the side's, as a share and a count."""

    def share(count: str, events: str) -> str:
        if int(events) == 0:
            return "no event"
        return f"{float(count) / int(events):.3f} ({float(count):g})"

    return table(
        (
            "Detector",
            "Arm",
            "Found @1 %",
            "Chance",
            "Excess",
            "Averaged",
            "Onset",
            "Short, excess",
            "Chosen",
        ),
        [
            (
                row["detector"],
                row["arm"],
                share(row["found"], row["events"]),
                share(row["chance"], row["events"]),
                share(row["excess"], row["events"]),
                share(row["averaged"], row["events"]),
                share(row["onset"], row["events"]),
                share(row["short_excess"], row["short_events"]),
                "yes" if row["chosen"] == "1" else "",
            )
            for row in rows
        ],
    )


def removed_from(directory: Path, section: str, left_out: LeftOut | None) -> dict[str, object]:
    """What a choice left out of ``section``: the classes, and the events and windows removed."""
    if left_out is None:
        return {}
    rows = next(iter(selection_answers(directory, section).values()))
    kept_rows = without_events(rows, left_out)
    events = left_out.events
    touched = {
        number
        for p in rows
        for number in (int(p.target), *left_out.overlaps(p.window))
        if number in events
    }
    return {
        "classes": sorted(left_out.classes),
        "events_removed": len(touched),
        "windows_removed": len(rows) - len(kept_rows),
    }


def choose_recall(
    directory: Path,
    out: Path,
    *,
    section: str = SELECT,
    rotations: int = ROTATIONS,
    trial: int | None = None,
    left_out: LeftOut | None = None,
) -> str:
    """Write the choice by events found to ``out``: the CSV, then its table and the chosen arm."""
    out.mkdir(parents=True, exist_ok=True)
    readings = recall_readings(
        directory, section=section, rotations=rotations, trial=trial, left_out=left_out
    )
    stem = f"recall-{section}{left_out.stem if left_out else ''}"
    if trial is not None:
        stem += f"-trial-{trial}"
    path = out / f"{stem}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECALL_COLUMNS)
        writer.writeheader()
        writer.writerows(recall_rows(readings))
    with path.open(newline="") as handle:
        text = recall_table(list(csv.DictReader(handle)))
    (out / f"{stem}.md").write_text(text + "\n")
    chosen = choose_by_recall(readings)
    record_of = {
        **chosen.__dict__,
        "excess": chosen.excess,
        "section": section,
        "left_out": removed_from(directory, section, left_out),
        "rotations": rotations,
        "trial": trial,
    }
    (out / f"{stem}-chosen.json").write_text(json.dumps(record_of, indent=2) + "\n")
    return text


def chosen_of(directory: Path) -> dict[tuple[str, str], str]:
    """The chosen arm of each family and detector, as ``choose`` wrote them.

    Raises:
        SystemExit: If the choice was not made.
    """
    path = directory / "chosen.json"
    if not path.exists():
        raise SystemExit("choose first: the refit reads the chosen arms")
    return {(c["family"], c["detector"]): c["arm"] for c in json.loads(path.read_text())}


def refit_arms(chosen: Mapping[tuple[str, str], str]) -> list[Arm]:
    """The arms the refit measures: every arm some family and detector chose, each once."""
    names = sorted(set(chosen.values()))
    return [Arm.parse(name) for name in names]


def control_of(family: str) -> tuple[str, str]:
    """The control a family is read against: the states' direction, or the cells' bar."""
    return (CELLS, FARTHEST) if family == LOCAL else (STATES, DIRECTION)


def paired(
    candidate: Mapping[int, AnomalyRanking], control: Mapping[int, AnomalyRanking]
) -> tuple[dict[int, AnomalyRanking], dict[int, AnomalyRanking]]:
    """The two sides over the same seeds: a seedless side stands beside every seed of the other."""
    if set(candidate) == {NO_SEED} and set(control) != {NO_SEED}:
        return dict.fromkeys(control, candidate[NO_SEED]), dict(control)
    if set(control) == {NO_SEED} and set(candidate) != {NO_SEED}:
        return dict(candidate), dict.fromkeys(candidate, control[NO_SEED])
    return dict(candidate), dict(control)


def reading_row(
    family: str,
    detector: str,
    arm: str,
    candidate: Mapping[int, AnomalyRanking],
    control: Mapping[int, AnomalyRanking],
    *,
    chosen: bool,
    bootstrap: PairedUnitBootstrap,
) -> ContextRow:
    """The row of one arm against its control, seeds paired; T is nought where neither side draws.

    Raises:
        SystemExit: If the control lacks a seed the arm holds.
    """
    candidate, control = paired(candidate, control)
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


def missing_refits(directory: Path, conditions: Mapping[str, Any]) -> list[str]:
    chosen = chosen_of(directory)
    wanted = [
        cell_name(REFIT, family, detector, arm, seed)
        for family, detector, arm, seed in arm_cells(REFIT, refit_arms(chosen), conditions["seeds"])
    ]
    wanted += [
        cell_name(REFIT, family, detector, None, seed)
        for family, detector, seed in control_cells(conditions["seeds"])
    ]
    held = done_cells(directory)
    return sorted(cell for cell in wanted if cell not in held)


def event_annotations(
    root: Path, mission: str, channels: Iterable[int]
) -> dict[int, tuple[str, str]]:
    """Each scored event's class and locality by its number, numbered as the ground truth numbers.

    Events touching no channel the corpus reads are left out and the gaps are not numbered, in the
    order the labels list them; the caller checks the classes against the ground truth's.

    Raises:
        ValueError: If a label names no channel, as the ground truth's reader refuses it.
    """
    folder = root / mission
    with (folder / "anomaly_types.csv").open(newline="") as handle:
        types = {row["ID"]: row for row in csv.DictReader(handle)}
    read = set(channels)
    order: list[str] = []
    with (folder / "labels.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            channel = row["Channel"]
            if not channel.startswith("channel_"):
                raise ValueError(f"{folder / 'labels.csv'}: {row['ID']} names no channel")
            if int(channel.removeprefix("channel_")) not in read:
                continue
            if row["ID"] not in order:
                order.append(row["ID"])
    annotations = {}
    number = 0
    for identifier in order:
        if types[identifier]["Category"] == "Communication Gap":
            continue
        number += 1
        annotations[number] = (types[identifier]["Class"], types[identifier]["Locality"])
    return annotations


def after_event_share(predictions: Sequence[WindowPrediction], hours: int, share: float) -> float:
    """Of the normal windows scored highest, the share within ``hours`` after an event's last."""
    ends = np.asarray([p.window.ends_at for p in predictions], dtype=np.float64)
    events = np.asarray([AnomalyScheme.is_event(p.target) for p in predictions], dtype=bool)
    scores = np.asarray([p.predicted for p in predictions])
    if not events.any() or events.all():
        return 0.0
    event_ends = np.sort(ends[events])
    top = int(share * (~events).sum())
    normal = np.flatnonzero(~events)
    highest = normal[np.argsort(-scores[normal], kind="stable")[:top]]
    if len(highest) == 0:
        return 0.0
    place = np.searchsorted(event_ends, ends[highest], side="left") - 1
    since = np.where(place >= 0, ends[highest] - event_ends[np.maximum(place, 0)], np.inf)
    return float(np.mean((since >= 1.0) & (since <= hours)))


def found_table(
    directory: Path,
    cells: Iterable[Mapping[str, str]],
    *,
    section: str,
    localities: Mapping[int, str],
    spans: Sequence[int],
) -> str:
    """Per cell: the events found at the share, the local ones, and those unfound before."""
    dropped = read_truncated(directory, section)
    rows = []
    for cell in sorted(cells, key=lambda c: c["cell"]):
        if cell["section"] != section:
            continue
        answers = kept(read_answers(directory / "answers" / f"{cell['cell']}.csv"), dropped)
        found = events_found(answers, SHARE)
        local = sum(localities.get(event) == "Local" for event in found)
        rows.append(
            (
                cell["cell"],
                str(len(found)),
                str(local),
                str(len(found & UNFOUND_BEFORE)),
                " ".join(str(e) for e in sorted(found & UNFOUND_BEFORE)),
                " ".join(f"{after_event_share(answers, span, SHARE):.2f}" for span in spans),
            )
        )
    return table(
        (
            "Cell",
            "Found",
            "Local",
            "Unfound before",
            "Which",
            " / ".join(f"after {span} h" for span in spans),
        ),
        rows,
    )


def read_tables(directory: Path, localities: Mapping[int, str], *, partial: bool = False) -> str:
    """Every table of the reading, written beside the cells and returned.

    Raises:
        SystemExit: If refit cells are missing and the reading is not marked partial.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    missing = missing_refits(directory, conditions)
    if missing and not partial:
        raise SystemExit(f"{len(missing)} refit cells are missing, the first {missing[0]}")
    chosen = chosen_of(directory)
    cells = cells_of(directory)
    rankings = rankings_of(directory, cells, section=REFIT)
    bootstrap = PairedUnitBootstrap(resamples=RESAMPLES, seed=0)
    rows = []
    for (family, detector), arms in sorted(rankings.items()):
        if family in (STATES, CELLS):
            continue
        control = rankings.get(control_of(family), {}).get("", {})
        if not control:
            continue
        for arm, seeds in sorted(arms.items()):
            rows.append(
                reading_row(
                    family,
                    detector,
                    arm,
                    seeds,
                    control,
                    chosen=chosen.get((family, detector)) == arm,
                    bootstrap=bootstrap,
                )
            )
    rows = corrected(rows)
    declared = [row for row in rows if (row.family, row.detector) in DECLARED]
    beside = [row for row in rows if (row.family, row.detector) not in DECLARED]
    controls = table(
        ("Family", "Detector", "Area per seed", "Area", "Recall @1 %"),
        [
            (
                family,
                detector,
                " ".join(f"{r.average_precision:.3f}" for _, r in sorted(seeds.items())),
                f"{fmean(r.average_precision for r in seeds.values()):.3f}",
                f"{fmean(r.event_recall_at_false_alarm_share(SHARE) for r in seeds.values()):.3f}",
            )
            for (family, detector), arms in sorted(rankings.items())
            if family in (STATES, CELLS)
            for seeds in [arms[""]]
        ],
    )
    dropped = read_truncated(directory, REFIT)
    text = "\n\n".join(
        [
            context_table(declared),
            context_table(beside),
            controls,
            f"Windows not read, their context of {conditions['widest_hours']} hours reaching "
            f"into a month the task never loaded: {len(dropped)} of "
            f"{conditions['windows_read']}, under every arm alike.",
            found_table(
                directory, cells, section=REFIT, localities=localities, spans=conditions["spans"]
            ),
        ]
    )
    (directory / "reading.md").write_text(text + "\n")
    return text


# The refitted control must rank the context report's windows this alike: Spearman's
# correlation at least this, and the top share of normal windows held whole.
LEAST_SPEARMAN = 0.9999


def agreement_of(a: Floats, b: Floats) -> tuple[float, float]:
    """Spearman's correlation of two scorings of the same windows, and the top share held.

    Tied windows share their mean rank, as Spearman's coefficient defines it: most hours depart
    by nothing and tie, and an order broken within a tie by the sort would read as disagreement.
    """
    spearman = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 1 else 1.0
    top = max(1, int(SHARE * len(a)))
    held = len(set(np.argsort(-a)[:top].tolist()) & set(np.argsort(-b)[:top].tolist())) / top
    return spearman, held


def rank_agreement(directory: Path, cell: str, against: Path, *, by_unit: bool = False) -> str:
    """How alike two cells rank the same windows: Spearman's correlation and the top share held.

    The standardisation's floor moves a score by a little, so the check reads ranks, not gaps;
    no area is read. Read by unit, the least agreement over the units is held to the declared
    least, so a cell whose alarm reads each unit alone can be checked against one whose alarm
    does not.

    Raises:
        SystemExit: If the two score different windows, or rank them less alike than the
            declared least.
    """
    own = {
        (str(p.window.unit), p.window.position): p.predicted
        for p in read_answers(directory / "answers" / f"{cell}.csv")
    }
    other = {(str(p.window.unit), p.window.position): p.predicted for p in read_answers(against)}
    if set(own) != set(other):
        raise SystemExit(f"{cell} and {against.name} score different windows")
    keys = sorted(own)
    groups = sorted({unit for unit, _ in keys}) if by_unit else [None]
    agreements = []
    for unit in groups:
        chosen = [k for k in keys if unit is None or k[0] == unit]
        agreements.append(
            agreement_of(
                np.asarray([own[k] for k in chosen]), np.asarray([other[k] for k in chosen])
            )
        )
    spearman = min(s for s, _ in agreements)
    held = min(h for _, h in agreements)
    how = f"least over {len(groups)} units" if by_unit else "over every window"
    text = (
        f"{cell} against {against.name}, {how}: Spearman {spearman:.5f}, "
        f"top {SHARE:.0%} held {held:.3f}"
    )
    if spearman < LEAST_SPEARMAN or held < 1.0:
        raise SystemExit(f"{text}: less alike than the declared least, the code is wrong")
    return text


def localities_of(task: str) -> dict[int, str]:
    """The locality of every scored event of the task's mission, checked against the ground truth.

    Raises:
        SystemExit: If the corpus is not downloaded or the numbering does not match the truth's.
    """
    return {number: locality for number, (_, locality) in annotations_of(task).items()}


def annotations_of(task: str) -> dict[int, tuple[str, str]]:
    """The class and locality of every scored event of the task's mission, checked.

    Raises:
        SystemExit: If the corpus is not downloaded or the numbering does not match the truth's.
    """
    return ground_truth_of(task)[1]


def ground_truth_of(task: str) -> tuple[EsaAdRawGroundTruth, dict[int, tuple[str, str]]]:
    """The task's ground truth and the class and locality of its mission's events, checked.

    Raises:
        SystemExit: If the corpus is not downloaded or the numbering does not match the truth's.
    """
    from emblema.catalog.adapters.readers.esa_ad_raw import EsaAdRawCorpusReader
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from scripts.raw_corpora import raw_root

    mission = KnownTasks.named(task).unit_prefix.rstrip("/")
    root = raw_root("esa_ad")
    if root is None:
        raise SystemExit("the ESA Anomaly Dataset is not downloaded")
    channels = EsaAdRawCorpusReader.POLICY.channels[mission]
    truth = EsaAdRawGroundTruth(
        root,
        channels=EsaAdRawCorpusReader.POLICY.channels,
        origin_seconds=ORIGIN_SECONDS,
        window_hours=KnownGroundTruths.ESA_AD_RAW_WINDOW_HOURS,
    )
    annotations = event_annotations(root, mission, channels)
    if {n: kind for n, (kind, _) in annotations.items()} != dict(truth.classes_of(mission)):
        raise SystemExit("the events numbered here are not the ground truth's")
    return truth, annotations


def early_sides(
    windows: Sequence[LabelledWindow], select_from: str
) -> tuple[list[LabelledWindow], list[LabelledWindow]]:
    """The fitting side's windows from ``select_from`` on, which fit, and those before, scored.

    Nothing is taught, so an event on both sides of the cut is scored where it lies before it.

    Raises:
        SystemExit: If the months before the cut hold no event or none lie from it on.
    """
    fit = [w for w in windows if month_of(w) >= select_from]
    scored = [w for w in windows if month_of(w) < select_from]
    if not fit or not any(AnomalyScheme.is_event(w.target) for w in scored):
        raise SystemExit(f"a cut at {select_from} leaves no event before it or nothing after it")
    return fit, scored


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every cell of a section, those already answered skipped; no area is written."""
    import time

    from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
    from emblema.evaluation.contracts.identifiers import CandidateRef

    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = "trial" if arguments.trial else committed_tree()
    refuse_another_commit(out / "conditions.json", commit)
    bench = bench_of(arguments)
    reader = Reader(bench, arguments.context)
    if arguments.section == EARLY:
        later, before = early_sides(bench.pool, arguments.select_from)
        fit = [reader.at[(str(w.window.unit), w.window.position)] for w in later]
        scored = [reader.at[(str(w.window.unit), w.window.position)] for w in before]
        taught: list[LabelledWindow] = []
    else:
        fit, scored, taught = reader.sides(arguments.section, arguments.select_from)
    widest = max(arguments.spans)
    truncated = truncated_windows(
        [reader.windows[i] for i in scored],
        {month_of(w) for w in reader.windows},
        widest,
        ORIGIN_SECONDS,
    )
    write_truncated(out / f"truncated-{arguments.section}.csv", truncated)
    conditions = {
        "commit": commit,
        "context": str(arguments.context),
        "features": list(arguments.features),
        "spans": list(arguments.spans),
        "seeds": list(arguments.seeds),
        "select_from": arguments.select_from,
        "events_in_pool": len(
            {int(w.target) for w in bench.pool if AnomalyScheme.is_event(w.target)}
        ),
        "windows_loaded": len(reader.windows),
        "windows_read": len(bench.read),
        "widest_hours": widest,
        "trial": bool(arguments.trial),
        f"windows_not_read_in_{arguments.section}": len(truncated),
    }
    path = out / "conditions.json"
    refuse_other_conditions(path, conditions)
    held = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**held, **conditions}, indent=2) + "\n")
    if arguments.section in CHOOSING:
        arms = arms_of(arguments.features, arguments.spans)
    else:
        arms = refit_arms(chosen_of(out))
    held_cells = done_cells(out)
    departures = Departures(reader.cells, out / "departures")
    units_fit = [str(reader.windows[i].window.unit) for i in fit]
    units_scored = [str(reader.windows[i].window.unit) for i in scored]
    contamination = (
        reader.catalogue.arm_of(CandidateRef("untrained_knn@distance=whitened,reference=unit"))
        .recipe(seed=1)
        .contamination
    )
    fit_array = np.asarray(fit, dtype=np.int64)
    scored_array = np.asarray(scored, dtype=np.int64)

    def keep(
        cell: str,
        family: str,
        detector: str,
        arm: str | None,
        seed: int,
        answers: Any,
        started: float,
    ) -> None:
        write_answers(out / "answers" / f"{cell}.csv", answers)
        if not (out / "cells.csv").exists():
            record(out / "cells.csv", CELL_COLUMNS)
        record(
            out / "cells.csv",
            (
                cell,
                arguments.section,
                family,
                detector,
                arm or "",
                seed,
                len({int(w.target) for w in taught}),
                len(fit),
                len(scored),
                f"{time.perf_counter() - started:.1f}",
            ),
        )
        print(f"{cell}: {time.perf_counter() - started:.0f} s", flush=True)

    def predictions(scores: Floats) -> tuple[WindowPrediction, ...]:
        return tuple(
            WindowPrediction(
                window=reader.windows[i].window, target=reader.windows[i].target, predicted=float(s)
            )
            for i, s in zip(scored, scores.tolist(), strict=True)
        )

    present = departures.present()
    states: dict[int, Floats] = (
        {}
        if arguments.section == EARLY
        else {seed: reader.states(CONTROL, seed, arguments.device) for seed in arguments.seeds}
    )
    for arm in arms:
        wanted = arm_cells(arguments.section, [arm], arguments.seeds)
        if arguments.section == EARLY:
            wanted = [cell for cell in wanted if cell[0] == LOCAL]
        if all(cell_name(arguments.section, *c) in held_cells for c in wanted):
            continue
        started = time.perf_counter()
        deviations = departures.of(arm, arguments.section, fit)
        print(
            f"{arguments.section}-{arm.name}: departures in {time.perf_counter() - started:.0f} s",
            flush=True,
        )
        for family, detector, name, seed in wanted:
            cell = cell_name(arguments.section, family, detector, name, seed)
            if cell in held_cells:
                continue
            started = time.perf_counter()
            if family == LOCAL:
                scores = window_scores(deviations, present, detector)
                alarm = ScoreAlarm.set_over(
                    scores[fit_array], contamination=contamination, units=units_fit
                )
                keep(
                    cell,
                    family,
                    detector,
                    name,
                    seed,
                    predictions(alarm.above(scores[scored_array], units=units_scored)),
                    started,
                )
                continue
            rows = deviations
            if detector == COMBINED:
                rows = np.concatenate((states[seed], deviations), axis=1)
            elif detector == TOP:
                rows = largest_departures(deviations)
            elif detector == ORDERED:
                rows = np.concatenate((states[seed], largest_departures(deviations)), axis=1)
            answers = reader.answers(
                TREES if detector == TREES else DIRECTION,
                rows[fit_array],
                rows[scored_array],
                fit,
                scored,
                taught,
                seed=seed if seed != NO_SEED else min(arguments.seeds),
            )
            keep(cell, family, detector, name, seed, answers, started)
    if arguments.section == REFIT or (arguments.controls and arguments.section == SELECT):
        for family, detector, seed in control_cells(arguments.seeds):
            cell = cell_name(arguments.section, family, detector, None, seed)
            if cell in held_cells:
                continue
            started = time.perf_counter()
            if family == STATES:
                rows = states[seed]
                answers = reader.answers(
                    detector, rows[fit_array], rows[scored_array], fit, scored, taught, seed=seed
                )
            else:
                rows_fit = reader.cell_rows(CONTROL, fit)
                rows_scored = reader.cell_rows(CONTROL, scored)
                answers = reader.answers(
                    FARTHEST, rows_fit, rows_scored, fit, scored, taught, seed=min(arguments.seeds)
                )
            keep(cell, family, detector, None, seed, answers, started)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="every cell of a section, no area")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True)
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument(
        "--weights", nargs=3, action="append", default=[], metavar=("NAME", "KEY", "CHECKSUM")
    )
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--context", type=Path, required=True, help="the context report's run")
    measuring.add_argument("--section", choices=SECTIONS, required=True)
    measuring.add_argument(
        "--features", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS)
    )
    measuring.add_argument("--spans", nargs="+", type=int, default=[6, 24, 72])
    measuring.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    measuring.add_argument("--select-from", required=True, help="the first month that chooses")
    measuring.add_argument("--controls", action="store_true", help="measure the controls too")
    measuring.add_argument("--device", default="cpu")
    measuring.add_argument("--trial", action="store_true")
    choosing = commands.add_parser("choose", help="the choice from the selection months")
    choosing.add_argument("--dir", type=Path, required=True)
    choosing.add_argument("--partial", action="store_true")
    by_recall = commands.add_parser(
        "choose-recall", help="the no-model arm finding the most events above chance"
    )
    by_recall.add_argument("--dir", type=Path, required=True)
    by_recall.add_argument("--section", choices=CHOOSING, required=True)
    by_recall.add_argument("--out", type=Path, required=True)
    by_recall.add_argument("--rotations", type=int, default=ROTATIONS)
    by_recall.add_argument("--task", help="the task whose event classes --leave-out names")
    by_recall.add_argument(
        "--leave-out", nargs="+", default=[], metavar="CLASS", help="event classes not scored"
    )
    by_recall.add_argument(
        "--trial", type=int, metavar="SEED", help="turn every month's scores first"
    )
    reading = commands.add_parser("read", help="the reading on the validation months")
    reading.add_argument("--dir", type=Path, required=True)
    reading.add_argument("--task", required=True)
    reading.add_argument("--task-id", required=True)
    reading.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    reading.add_argument(
        "--weights", nargs=3, action="append", default=[], metavar=("NAME", "KEY", "CHECKSUM")
    )
    reading.add_argument("--partial", action="store_true")
    reading.add_argument("--device", default="cpu")
    checking = commands.add_parser("check", help="a cell's answers against another's, no area")
    checking.add_argument("--dir", type=Path, required=True)
    checking.add_argument("--cell", required=True)
    checking.add_argument("--against", type=Path, required=True)
    checking.add_argument("--by-unit", action="store_true", help="the least agreement over units")
    return parser.parse_args(argv)


def classes_left_out(task: str | None, kinds: Sequence[str]) -> LeftOut | None:
    """The events of each class named and the task's overlaps, or nothing when none is named.

    Raises:
        SystemExit: If classes are named without a task, or a class has no event.
    """
    if not kinds:
        return None
    if task is None:
        raise SystemExit("--leave-out names classes of a task: name it with --task")
    truth, annotations = ground_truth_of(task)
    events = {
        kind: sorted(n for n, (each, _) in annotations.items() if each == kind) for kind in kinds
    }
    empty = [kind for kind, numbers in events.items() if not numbers]
    if empty:
        raise SystemExit(f"no event of {', '.join(empty)} in {task}")
    return LeftOut(classes=events, overlaps=truth.events_of)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.command == "measure":
        measure(arguments)
    elif arguments.command == "choose":
        choices = choose_from(arguments.dir, partial=arguments.partial)
        (arguments.dir / "chosen.json").write_text(
            json.dumps([choice.__dict__ for choice in choices], indent=2) + "\n"
        )
        text = choice_table(choices)
        (arguments.dir / "chosen.md").write_text(text + "\n")
        print(text)
    elif arguments.command == "choose-recall":
        print(
            choose_recall(
                arguments.dir,
                arguments.out,
                section=arguments.section,
                rotations=arguments.rotations,
                trial=arguments.trial,
                left_out=classes_left_out(arguments.task, arguments.leave_out),
            )
        )
    elif arguments.command == "read":
        print(read_tables(arguments.dir, localities_of(arguments.task), partial=arguments.partial))
    else:
        print(check_cell(arguments.dir, arguments.cell, arguments.against))
        print(
            rank_agreement(
                arguments.dir, arguments.cell, arguments.against, by_unit=arguments.by_unit
            )
        )


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
