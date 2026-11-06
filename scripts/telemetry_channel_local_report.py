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

Three commands, so a reading stays blind until the choice is made:

- ``measure``: the departures of every loaded window under each span and feature set, kept
  on disk, and every cell's answers of a section (``select`` chooses on the later fitting
  months, ``refit`` reads the validation months); no area is written.
- ``choose``: for each family and detector, the arm with the highest mean area on the
  selection months; within 0.01 of it the shorter span, then the hour's summary before the
  bins.
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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
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

SECTIONS = (SELECT, REFIT)
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


def rank_agreement(directory: Path, cell: str, against: Path) -> str:
    """How alike two cells rank the same windows: Spearman's correlation and the top share held.

    The standardisation's floor moves a score by a little, so the check reads ranks, not gaps;
    no area is read.

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
    a = np.asarray([own[k] for k in keys])
    b = np.asarray([other[k] for k in keys])
    ranks_a, ranks_b = a.argsort().argsort(), b.argsort().argsort()
    spearman = float(np.corrcoef(ranks_a, ranks_b)[0, 1]) if len(keys) > 1 else 1.0
    top = max(1, int(SHARE * len(keys)))
    held = len(set(np.argsort(-a)[:top].tolist()) & set(np.argsort(-b)[:top].tolist())) / top
    text = (
        f"{cell} against {against.name}: Spearman {spearman:.5f}, top {SHARE:.0%} held {held:.3f}"
    )
    if spearman < LEAST_SPEARMAN or held < 1.0:
        raise SystemExit(f"{text}: less alike than the declared least, the code is wrong")
    return text


def localities_of(task: str) -> dict[int, str]:
    """The locality of every scored event of the task's mission, checked against the ground truth.

    Raises:
        SystemExit: If the corpus is not downloaded or the numbering does not match the truth's.
    """
    from emblema.catalog.adapters.readers.esa_ad_raw import EsaAdRawCorpusReader
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
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
    return {number: locality for number, (_, locality) in annotations.items()}


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
    if arguments.section == SELECT:
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
    states = {seed: reader.states(CONTROL, seed, arguments.device) for seed in arguments.seeds}
    for arm in arms:
        wanted = arm_cells(arguments.section, [arm], arguments.seeds)
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
    if arguments.section == REFIT or arguments.controls:
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
    return parser.parse_args(argv)


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
    elif arguments.command == "read":
        print(read_tables(arguments.dir, localities_of(arguments.task), partial=arguments.partial))
    else:
        print(check_cell(arguments.dir, arguments.cell, arguments.against))
        print(rank_agreement(arguments.dir, arguments.cell, arguments.against))


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
