"""Choose what a telemetry pretext's window holds, from the fitting months and with no label.

A pretext that hides a window's last hour and asks for it from the hours before needs a window
length and a cell resolution. No model is needed to ask what history can tell about the hidden
hour: a ridge forecast of the hour's summary from the cells before it, against the median of the
five hours before it (the package's channel-local detector, ADR-0075). The ridge forecasts the
hour's departure from that median, from inputs with the level taken out, so a forecast shrunk to
nothing is the median itself and a level drifting between months is not extrapolated. Its skill
under each resolution and length shows where more history or finer cells stop helping, and how
much a forecast can gain over the median at all. No label is read; the targets are what the
hidden hour held.

Four commands:

- ``measure``: on the fitting months, the forecasts' errors per month for every resolution,
  length and set of inputs; each channel's hourly autocorrelation; each channel's monthly
  median error; the fitting windows per month. CSV only.
- ``cost``: the small shape's forward and backward pass over a window of each layout, on a
  device, timed, with the memory it held.
- ``pool``: the rule over the union of the missions' runs, the corpus a pretraining reads.
- ``read``: the skills with intervals over the months, the checks that the measure is believable,
  the layout the rule chooses, whether the errors carry a yearly structure, and how many months
  the held-out block takes.

The cells come from the context report's run, laid once from the published one-hour windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.arrays.joined_cells import JoinedCells  # noqa: E402
from scripts.reporting import table  # noqa: E402

Floats = NDArray[np.float64]

PRESENCE = BinnedTokens.PRESENCE
LOG_COUNT = BinnedTokens.LOG_COUNT
MEAN, MINIMUM, MAXIMUM, FIRST, LAST, OFFSET, SINCE = range(2, 9)
# The hidden hour's summary, as the channel-local detector reads it.
TARGETS = (MEAN, MINIMUM, MAXIMUM, LAST, LOG_COUNT)
TARGET_NAMES = ("mean", "minimum", "maximum", "last", "log_count")
# Bins per hour: one-hour, half-hour and ten-minute cells, all whole published cells.
RESOLUTIONS: Mapping[str, int] = {"60min": 1, "30min": 2, "10min": 6}
LENGTHS = (6, 12, 24, 48)
# The forecast's inputs: the channel's own history, or that beside every channel's last hour.
OWN, CROSS = "own", "cross"
VARIANTS = (OWN, CROSS)
# The rule reads the inputs the encoder sees: every channel.
RULED = CROSS
# The bar a forecast is measured against: the median of the five hours before (span 6).
BASELINE_HOURS = 5
# The first two thirds of the fitting months by time fit the forecasts; the rest are read.
TEACH_SHARE = 2 / 3
# Within the teaching months, the last quarter chooses each forecast's penalty.
INNER_SHARE = 3 / 4
# A penalty is a share of the rows fitted, since a standardised column's square sums to about
# that many: the same share shrinks alike on a quarter more rows when the forecast is refitted.
# The grid ends in a forecast of nothing, the median's answer itself: a finite penalty shrinks
# a window of thousands of correlated columns too little along its widest directions.
PENALTIES = (*(10.0**power for power in range(-6, 4)), math.inf)
# Readings beyond these quantiles of the teaching months are clipped, as a few of the
# satellites' readings lie thousands of deviations out.
CLIP = (0.001, 0.999)
DRAWS = 10_000
SEED = 1
LAGS = 72
# A yearly lag is read only over three years of months or more.
YEARLY_LAG, LEAST_YEARS = 12, 3
# The least hours a held-out block holds: a median's standard error is about 1.25 / sqrt(n) of
# the readings' spread for normal readings, under 3 % of it at 2,000 hours.
LEAST_BLOCK_HOURS = 2_000
LEAST_BLOCK_MONTHS = 3
YEARLY_BLOCK_MONTHS = 12
# A target's scale is at least this share of its own spread on the teaching rows, as the
# channel-local detector floors a history's spread (ADR-0075): a median that is nearly exact on
# a stepped target would otherwise weigh that target's every error hundreds of times over.
FLOOR_SHARE = 0.1
# The measure is believable when no arm's forecast loses to the median on the rows it was fitted
# on by more than this share, and none read loses more than the median's whole error. The grid
# holds a forecast of about nothing, so a worse one fights the measure rather than lacks signal.
LEAST_TEACHING_SKILL = -0.05
LEAST_READ_SKILL = -1.0
TEACH, READ = "teach", "read"
PENALTY_COLUMNS = ("resolution", "length", "variant", "channel", "target", "share")
ERROR_COLUMNS = (
    "side",
    "resolution",
    "length",
    "variant",
    "month",
    "forecast",
    "median",
    "cells",
)


@dataclass(frozen=True, order=True)
class Arm:
    """One window under one cell resolution and one set of forecast inputs.

    Attributes:
        resolution: The cells' width, a key of ``RESOLUTIONS``.
        length: Hours the window spans, the hidden one included.
        variant: The forecast's inputs, own history or with every channel's last hour.
    """

    resolution: str
    length: int
    variant: str

    @property
    def bins_per_hour(self) -> int:
        return RESOLUTIONS[self.resolution]

    @property
    def history_bins(self) -> int:
        return (self.length - 1) * self.bins_per_hour

    @property
    def tokens(self) -> int:
        """Cells per channel over the whole window, the hidden hour included."""
        return self.length * self.bins_per_hour


def arms() -> list[Arm]:
    return [Arm(r, length, v) for r in RESOLUTIONS for length in LENGTHS for v in VARIANTS]


def hour_summaries(published: Floats) -> Floats:
    """``[..., 9]``: each hour's one-hour cell from its ``[..., BASE_BINS, 9]`` published cells."""
    summary: Floats = JoinedCells(1).join(published)[..., 0, :]
    return summary


def gathered(series: Floats, places: NDArray[np.int64]) -> Floats:
    """``series[places]`` with a place of ``-1`` as cells of nothing."""
    found = series[np.maximum(places, 0)]
    nothing = (places < 0).reshape(places.shape + (1,) * (series.ndim - 1))
    out: Floats = np.where(nothing, 0.0, found)
    return out


def history_cells(channel: Floats, before: NDArray[np.int64], arm: Arm) -> Floats:
    """``[n, history_bins * 9]``: the cells of the hours before each window under ``arm``.

    ``channel`` is one channel's ``[windows, BASE_BINS, 9]`` published cells, ``before`` the
    ``[n, hours]`` loaded windows of the hours before each, earliest first, ``-1`` where none.
    The hours are laid out as one window's bins would be, carrying the last reading forward.
    """
    places = before[:, before.shape[1] - (arm.length - 1) :]
    published = gathered(channel, places)
    n, hours, bins, features = published.shape
    flat = published.reshape(n, hours * bins, features)
    laid = JoinedCells(arm.history_bins).join(flat)
    out: Floats = laid.reshape(n, -1)
    return out


def level_free(cells: Floats, level: Floats) -> Floats:
    """``[n, bins, 9]`` history cells with each row's level taken out.

    ``level`` is ``[n, 2]``: a reading's level, taken from the five values, and the log of a
    count, taken from the log of the count. A bin before the history's first reading holds
    nothing, so it stays zero rather than becoming the level's negative; a bin without a reading
    holds no count, so its log of the count stays zero too.
    """
    out = cells.copy()
    held = np.maximum.accumulate(cells[..., PRESENCE] > 0.0, axis=1)
    values = [MEAN, MINIMUM, MAXIMUM, FIRST, LAST]
    out[..., values] = np.where(held[..., None], cells[..., values] - level[:, None, 0, None], 0.0)
    # A bin without a reading has no count to compare; its presence already says so.
    present = cells[..., PRESENCE] > 0.0
    out[..., LOG_COUNT] = np.where(present, cells[..., LOG_COUNT] - level[:, None, 1], 0.0)
    return out


def departures(last: Floats, median: Floats) -> Floats:
    """``[n, channels, 6]``: each channel's last hour less its median, and whether it was held.

    ``last`` is ``[n, channels, 9]`` one-hour cells, ``median`` ``[n, channels, 5]`` the medians
    of the targets over the hours before; an hour or a median missing departs by nothing.
    """
    held = (last[..., PRESENCE] > 0.0) & ~np.isnan(median).any(axis=-1)
    away = np.where(held[..., None], last[..., list(TARGETS)] - np.nan_to_num(median), 0.0)
    out: Floats = np.concatenate((away, held[..., None].astype(np.float64)), axis=-1)
    return out


def floored_scale(bar_error: Floats, targets: Floats) -> Floats:
    """Each target's scale: the bar's mean error, at least ``FLOOR_SHARE`` of its own spread.

    Both are read over the same teaching rows; a target with no spread there keeps a scale of
    zero and is left out.
    """
    spread = np.abs(targets - targets.mean(axis=0)).mean(axis=0)
    out: Floats = np.where(spread > 0.0, np.maximum(bar_error, FLOOR_SHARE * spread), 0.0)
    return out


def median_of_hours(hours: Floats) -> tuple[Floats, NDArray[np.int64]]:
    """The median of each target over the hours held, and how many hours held a reading.

    ``hours`` is ``[n, k, 9]`` one-hour cells, an hour of nothing having no presence.
    """
    present = hours[..., PRESENCE] > 0.0
    values = np.where(present[..., None], hours[..., list(TARGETS)], np.nan)
    # A row with no hour held has no median; it is left out of the read rows by the caller.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median: Floats = np.nanmedian(values, axis=1)
    return median, present.sum(axis=1)


def clipped_hours(hours: Floats, teaching: Floats) -> Floats:
    """``[n, 9]`` one-hour cells with the targets clipped at the ``teaching`` rows' quantiles.

    The bar reads the hours before as the forecast's target reads the hidden one.
    """
    low, high = np.quantile(teaching[:, list(TARGETS)], CLIP, axis=0)
    out = hours.copy()
    out[:, list(TARGETS)] = np.clip(hours[:, list(TARGETS)], low, high)
    return out


def monthly_median_errors(
    median: Floats,
    targets: Floats,
    scale: Floats,
    rows: NDArray[np.bool_],
    months: Sequence[str],
    order: Sequence[str],
) -> list[tuple[str, float]]:
    """Each month's median, over its read rows, of the bar's mean scaled error over the targets.

    A target that never moved on the teaching rows is left out, as in the errors; a month with
    no row read is ``nan``.
    """
    moving = scale > 0.0
    scaled = (np.abs(median - targets)[:, moving] / scale[moving]).mean(axis=1)
    named = np.asarray(months)
    out = []
    for month in order:
        mine = rows & (named == month)
        out.append((month, float(np.median(scaled[mine])) if mine.any() else math.nan))
    return out


def enough_history(held: NDArray[np.int64], length: int) -> NDArray[np.bool_]:
    """Whether at least half of the ``length - 1`` hours before held a reading."""
    out: NDArray[np.bool_] = held >= math.ceil((length - 1) / 2)
    return out


def clipped(values: Floats, teach: NDArray[np.bool_]) -> Floats:
    """Each column clipped to the teaching rows' quantiles in ``CLIP``."""
    low, high = np.quantile(values[teach], CLIP, axis=0)
    out: Floats = np.clip(values, low, high)
    return out


def standardised(values: Floats, teach: NDArray[np.bool_]) -> tuple[Floats, NDArray[np.bool_]]:
    """Columns scaled by the teaching rows' mean and spread, those without spread dropped.

    Returns the scaled columns and which of ``values``' columns they are.
    """
    mean = values[teach].mean(axis=0)
    spread = values[teach].std(axis=0)
    kept: NDArray[np.bool_] = spread > 1e-12
    out: Floats = (values[:, kept] - mean[kept]) / spread[kept]
    return out, kept


@dataclass(frozen=True)
class Moments:
    """The sums a ridge forecast is solved from, over some rows: added, they are the union's.

    Attributes:
        rows: How many rows the sums hold.
        gram: The inputs' products, ``x.T @ x``.
        cross: The inputs' products with the targets, ``x.T @ y``.
    """

    rows: int
    gram: Floats
    cross: Floats

    @classmethod
    def of(cls, inputs: Floats, targets: Floats) -> Moments:
        return cls(rows=len(inputs), gram=inputs.T @ inputs, cross=inputs.T @ targets)

    def __add__(self, other: Moments) -> Moments:
        return Moments(
            rows=self.rows + other.rows,
            gram=self.gram + other.gram,
            cross=self.cross + other.cross,
        )

    def solved(self, columns: NDArray[np.intp], penalties: Sequence[Floats]) -> list[Floats]:
        """The weights over ``columns`` under each array of per-target penalties.

        A penalty is a share of the rows; an infinite one forecasts nothing. The targets are not
        centred and every column is penalised, the constant one included, so the forecast
        shrinks towards nothing, the median's answer.
        """
        values, vectors = np.linalg.eigh(self.gram[np.ix_(columns, columns)])
        projected = vectors.T @ self.cross[columns]
        return [
            vectors @ (projected / (values[:, None] + self.rows * penalty[None, :]))
            for penalty in penalties
        ]


def chosen_penalty(
    inner: Moments, held_inputs: Floats, held_targets: Floats, columns: NDArray[np.intp]
) -> Floats:
    """Per target, the penalty whose forecast solved on ``inner`` errs least on the held rows."""
    targets = held_targets.shape[1]
    grid = [np.full(targets, penalty) for penalty in PENALTIES]
    errors = [
        np.abs(held_inputs[:, columns] @ weights - held_targets).mean(axis=0)
        for weights in inner.solved(columns, grid)
    ]
    best: Floats = np.asarray(PENALTIES)[np.argmin(np.stack(errors), axis=0)]
    return best


def forecast_of(
    moments: Moments, inputs: Floats, columns: NDArray[np.intp], penalties: Floats
) -> Floats:
    """Every row's forecast of each target, solved on ``moments`` under its own penalty."""
    (weights,) = moments.solved(columns, [penalties])
    out: Floats = inputs[:, columns] @ weights
    return out


@dataclass(frozen=True)
class MonthErrors:
    """One arm's scaled absolute errors summed per read month, beside the median's."""

    forecast: Mapping[str, float]
    median: Mapping[str, float]
    cells: Mapping[str, int]


def month_errors(
    forecast: Floats,
    median: Floats,
    targets: Floats,
    scale: Floats,
    rows: NDArray[np.bool_],
    months: Sequence[str],
) -> MonthErrors:
    """Errors of the ``rows`` read, each target over its ``scale``, summed per month."""
    usable = scale > 0.0
    f = (np.abs(forecast - targets)[:, usable] / scale[usable]).sum(axis=1)
    m = (np.abs(median - targets)[:, usable] / scale[usable]).sum(axis=1)
    names, of_row = np.unique(np.asarray(months), return_inverse=True)
    picked = np.flatnonzero(rows)
    per_f = np.bincount(of_row[picked], weights=f[picked], minlength=len(names))
    per_m = np.bincount(of_row[picked], weights=m[picked], minlength=len(names))
    per_n = np.bincount(of_row[picked], minlength=len(names)) * int(usable.sum())
    held = per_n > 0
    return MonthErrors(
        {str(n): float(v) for n, v, h in zip(names, per_f, held, strict=True) if h},
        {str(n): float(v) for n, v, h in zip(names, per_m, held, strict=True) if h},
        {str(n): int(v) for n, v, h in zip(names, per_n, held, strict=True) if h},
    )


def autocorrelation(series: Floats, lags: int) -> Floats:
    """The autocorrelation at lags 1 to ``lags`` over the pairs both held, ``nan`` for none."""
    held = ~np.isnan(series)
    centred = np.where(held, series - np.nanmean(series), 0.0)
    spread = float(np.sqrt((centred[held] ** 2).mean())) if held.any() else 0.0
    out = np.full(lags, np.nan)
    if spread == 0.0:
        return out
    for lag in range(1, lags + 1):
        both = held[lag:] & held[:-lag]
        if both.sum() > 2:
            out[lag - 1] = float((centred[lag:] * centred[:-lag])[both].mean()) / spread**2
    return out


def detrended(values: Floats) -> Floats:
    """``values`` less their least-squares line over position, ``nan`` kept."""
    places = np.arange(len(values), dtype=np.float64)
    held = ~np.isnan(values)
    if held.sum() < 2:
        return values.copy()
    slope, intercept = np.polyfit(places[held], values[held], 1)
    out: Floats = values - (slope * places + intercept)
    return out


@dataclass(frozen=True)
class Yearly:
    """Whether the channels' monthly errors repeat after a year.

    Attributes:
        read: Whether there are enough months to ask.
        lag_twelve: The median over the channels of the lag-12 autocorrelation.
        band: The two-sided 95 % band of white noise, 1.96 / sqrt(months).
    """

    read: bool
    lag_twelve: float
    band: float

    @property
    def present(self) -> bool:
        return self.read and self.lag_twelve > self.band


def yearly(monthly: Mapping[int, Floats]) -> Yearly:
    """The yearly structure of each channel's monthly median errors, detrended, months in order."""
    months = max((len(v) for v in monthly.values()), default=0)
    band = 1.96 / math.sqrt(months) if months else math.inf
    if months < YEARLY_LAG * LEAST_YEARS:
        return Yearly(read=False, lag_twelve=math.nan, band=band)
    lags = [autocorrelation(detrended(v), YEARLY_LAG)[-1] for v in monthly.values()]
    held = [lag for lag in lags if not math.isnan(lag)]
    return Yearly(read=True, lag_twelve=float(np.median(held)) if held else math.nan, band=band)


def block_months(windows_per_month: Sequence[int], *, yearly_present: bool) -> int:
    """How many of the last fitting months the held-out block takes.

    At least ``LEAST_BLOCK_MONTHS`` and enough months, counted back from the last, to hold
    ``LEAST_BLOCK_HOURS``; a year when the errors repeat after one, so the block does not stand
    for one season.

    Raises:
        ValueError: If the months hold fewer hours than a block needs.
    """
    held, months = 0, 0
    for count in reversed(windows_per_month):
        held += count
        months += 1
        if months >= LEAST_BLOCK_MONTHS and held >= LEAST_BLOCK_HOURS:
            break
    else:
        raise ValueError(f"{len(windows_per_month)} months hold {held} hours, short of a block")
    if yearly_present:
        months = max(months, YEARLY_BLOCK_MONTHS)
    if months >= len(windows_per_month):
        raise ValueError(f"a block of {months} months leaves no month to pretrain on")
    return months


@dataclass(frozen=True)
class Skill:
    """An arm's skill over the median, ``1 - forecast error / median error``, with an interval."""

    arm: Arm
    skill: float
    low: float
    high: float


def month_draws(months: int, draws: int, seed: int) -> NDArray[np.float64]:
    """``[draws, months]``: how often each month is drawn under a bootstrap over months."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, months, size=(draws, months))
    counts = np.zeros((draws, months))
    np.add.at(counts, (np.arange(draws)[:, None], picks), 1.0)
    return counts


def skills(
    errors: Mapping[Arm, MonthErrors], draws: NDArray[np.float64], months: Sequence[str]
) -> tuple[dict[Arm, Skill], dict[Arm, Floats]]:
    """Every arm's skill and its interval, and the skill under each draw for paired reads.

    The median's error is the same under every arm, so the arms differ in their forecasts only.
    """
    drawn: dict[Arm, Floats] = {}
    out: dict[Arm, Skill] = {}
    for arm, each in errors.items():
        f = np.asarray([each.forecast.get(m, 0.0) for m in months])
        m = np.asarray([each.median.get(m, 0.0) for m in months])
        point = 1.0 - f.sum() / m.sum()
        under = 1.0 - (draws @ f) / (draws @ m)
        drawn[arm] = under
        low, high = np.quantile(under, (0.025, 0.975))
        out[arm] = Skill(arm, float(point), float(low), float(high))
    return out, drawn


def better(drawn: Mapping[Arm, Floats], than: Arm, arm: Arm) -> bool:
    """Whether ``arm`` forecasts better than ``than``: the paired difference's low edge above 0."""
    return bool(np.quantile(drawn[arm] - drawn[than], 0.025) > 0.0)


def chosen_length(drawn: Mapping[Arm, Floats], resolution: str, variant: str) -> int:
    """The shortest length that no longer one forecasts better than."""
    for length in LENGTHS:
        here = Arm(resolution, length, variant)
        longer = [Arm(resolution, other, variant) for other in LENGTHS if other > length]
        if not any(better(drawn, here, each) for each in longer):
            return length
    return LENGTHS[-1]


def chosen_arm(drawn: Mapping[Arm, Floats], variant: str = RULED) -> Arm:
    """The coarsest resolution, at its chosen length, that no finer one forecasts better than."""
    order = sorted(RESOLUTIONS, key=lambda r: RESOLUTIONS[r])
    at = {r: Arm(r, chosen_length(drawn, r, variant), variant) for r in order}
    for place, resolution in enumerate(order):
        finer = order[place + 1 :]
        if not any(better(drawn, at[resolution], at[f]) for f in finer):
            return at[resolution]
    return at[order[-1]]


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def errors_of(rows: Sequence[Mapping[str, str]], side: str = READ) -> dict[Arm, MonthErrors]:
    """The per-month errors ``measure`` wrote on one side, summed over the channels, by arm."""
    forecast: dict[Arm, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    median: dict[Arm, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cells: dict[Arm, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["side"] != side:
            continue
        arm = Arm(row["resolution"], int(row["length"]), row["variant"])
        forecast[arm][row["month"]] += float(row["forecast"])
        median[arm][row["month"]] += float(row["median"])
        cells[arm][row["month"]] += int(row["cells"])
    return {a: MonthErrors(dict(forecast[a]), dict(median[a]), dict(cells[a])) for a in forecast}


def skill_table(found: Mapping[Arm, Skill], chosen: Arm) -> str:
    rows = [
        [
            s.arm.resolution,
            str(s.arm.length),
            s.arm.variant,
            str(s.arm.tokens),
            f"{s.skill:.3f} [{s.low:.3f}; {s.high:.3f}]",
            "chosen" if s.arm == chosen else "",
        ]
        for s in sorted(
            found.values(),
            key=lambda s: (s.arm.variant, RESOLUTIONS[s.arm.resolution], s.arm.length),
        )
    ]
    return table(
        ["Resolution", "Hours", "Inputs", "Cells a channel", "Skill over the median", ""], rows
    )


@dataclass(frozen=True)
class Believable:
    """Whether the measure reads as a measure, before any choice it makes is taken.

    Attributes:
        least_teaching: The lowest skill of any arm on the rows it was fitted on.
        least_read: The lowest skill of any arm on the rows read.
    """

    least_teaching: float
    least_read: float

    @property
    def holds(self) -> bool:
        return self.least_teaching >= LEAST_TEACHING_SKILL and self.least_read >= LEAST_READ_SKILL


def point_skills(errors: Mapping[Arm, MonthErrors]) -> dict[Arm, float]:
    return {
        arm: 1.0 - sum(each.forecast.values()) / sum(each.median.values())
        for arm, each in errors.items()
    }


def read(arguments: argparse.Namespace) -> None:
    """The tables and the choices from what ``measure`` and ``cost`` wrote."""
    out: Path = arguments.dir
    written = read_rows(out / "errors.csv")
    if not written or "side" not in written[0]:
        raise SystemExit(
            f"{out / 'errors.csv'} holds no side: written before the measure's correction"
        )
    errors = errors_of(written)
    months = sorted({m for e in errors.values() for m in e.median})
    draws = month_draws(len(months), DRAWS, SEED)
    found, drawn = skills(errors, draws, months)
    believable = Believable(
        least_teaching=min(point_skills(errors_of(written, TEACH)).values()),
        least_read=min(s.skill for s in found.values()),
    )
    chosen = chosen_arm(drawn)
    monthly: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for row in read_rows(out / "monthly.csv"):
        value = float(row["median_error"]) if row["median_error"] else math.nan
        monthly[int(row["channel"])].append((row["month"], value))
    series = {c: np.asarray([v for _, v in sorted(rows)]) for c, rows in monthly.items()}
    year = yearly(series)
    counts = [
        int(r["windows"]) for r in sorted(read_rows(out / "windows.csv"), key=lambda r: r["month"])
    ]
    block = block_months(counts, yearly_present=year.present)
    shares = [float(r["share"]) for r in read_rows(out / "penalties.csv")]
    low = sum(s == PENALTIES[0] for s in shares) / len(shares) if shares else 0.0
    high = sum(s == PENALTIES[-1] for s in shares) / len(shares) if shares else 0.0
    choice = {
        "believable": believable.holds,
        "least_teaching_skill": believable.least_teaching,
        "least_read_skill": believable.least_read,
        "penalties_at_the_least": low,
        "penalties_at_the_most": high,
        # An unbelievable measure chooses nothing, so nothing downstream can take its layout.
        "resolution": chosen.resolution if believable.holds else None,
        "length": chosen.length if believable.holds else None,
        "bins": chosen.tokens if believable.holds else None,
        "skill": found[chosen].skill if believable.holds else None,
        "yearly_read": year.read,
        "yearly_lag_twelve": year.lag_twelve,
        "yearly_band": year.band,
        "block_months": block if believable.holds else None,
        "block_hours": sum(counts[-block:]) if believable.holds else None,
    }
    (out / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(skill_table(found, chosen))
    print()
    print(json.dumps(choice, indent=2))
    if not believable.holds:
        print("the measure is not believable: a configuration fault, no choice taken", flush=True)


def pooled_rows(named: Mapping[str, Sequence[Mapping[str, str]]]) -> list[dict[str, str]]:
    """The errors of several runs as one corpus's, each month named by its run.

    A pretraining reads every mission at once, so its layout is the rule's over their union; a
    month of one mission is not the same month of another, so the months are drawn apart.
    """
    return [
        {**row, "month": f"{name}:{row['month']}"} for name, rows in named.items() for row in rows
    ]


def pool(arguments: argparse.Namespace) -> None:
    """The rule over the union of the runs ``measure`` wrote, one per mission."""
    if len(arguments.dirs) != len(arguments.names):
        raise SystemExit("every run needs one name")
    written = pooled_rows(
        {
            n: read_rows(d / "errors.csv")
            for n, d in zip(arguments.names, arguments.dirs, strict=True)
        }
    )
    errors = errors_of(written)
    months = sorted({m for e in errors.values() for m in e.median})
    found, drawn = skills(errors, month_draws(len(months), DRAWS, SEED), months)
    believable = Believable(
        least_teaching=min(point_skills(errors_of(written, TEACH)).values()),
        least_read=min(s.skill for s in found.values()),
    )
    chosen = chosen_arm(drawn)
    choice = {
        "runs": dict(zip(arguments.names, map(str, arguments.dirs), strict=True)),
        "months": len(months),
        "believable": believable.holds,
        "resolution": chosen.resolution if believable.holds else None,
        "length": chosen.length if believable.holds else None,
        "bins": chosen.tokens if believable.holds else None,
        "skill": found[chosen].skill if believable.holds else None,
    }
    arguments.out.write_text(json.dumps(choice, indent=2) + "\n")
    print(skill_table(found, chosen))
    print()
    print(json.dumps(choice, indent=2))


def cost(arguments: argparse.Namespace) -> None:  # pragma: no cover - device
    """Forward and backward over one window of each layout, timed, on a device."""
    import torch

    from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
    from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile

    shape = ExperimentFile.load(arguments.experiment).configuration().architecture
    device = torch.device(arguments.device)
    rows = []
    for resolution, per_hour in RESOLUTIONS.items():
        for length in LENGTHS:
            tokens = arguments.channels * length * per_hour
            for batch in arguments.batches:
                torch.manual_seed(SEED)
                laid = replace(shape, bins=length * per_hour)
                # Composed as the pretext over cells composes it (ObjectiveShape.build).
                encoder = SetEncoder.for_vocabulary(
                    laid,
                    arguments.channels,
                    value_embedding=torch.nn.Linear(laid.input_features, laid.width),
                ).to(device)
                features = torch.randn(batch, tokens, BinnedTokens.N_FEATURES, device=device)
                ids = torch.arange(tokens, device=device).remainder(arguments.channels)
                ids = ids.expand(batch, tokens)
                times = torch.rand(batch, tokens, device=device)
                timeless = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
                padding = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
                seconds, failed, started = math.nan, "", time.perf_counter()
                # A kernel that lays the attention out whole holds this much; past the budget
                # the step is not tried, so a host without such room is not pushed into swap.
                laid_out = batch * shape.heads * tokens**2 * 4
                if laid_out > arguments.budget_gb * 1e9:
                    failed = f"skipped: attention laid out whole holds {laid_out / 1e9:.0f} GB"
                    rows.append(cost_row(arguments.device, resolution, length, tokens, batch))
                    rows[-1].update({"failed": failed})
                    print(rows[-1], flush=True)
                    continue
                try:
                    for step in range(arguments.repeats + 1):
                        if step == 1:
                            synchronised(device)
                            started = time.perf_counter()
                        encoder(features, ids, times, timeless, padding).sum().backward()
                    synchronised(device)
                    seconds = (time.perf_counter() - started) / arguments.repeats
                except RuntimeError as error:
                    # Running out of memory is a measurement here; anything else is a fault.
                    if "out of memory" not in str(error).lower():
                        raise
                    failed = str(error).splitlines()[0][:120]
                rows.append(cost_row(arguments.device, resolution, length, tokens, batch))
                rows[-1].update(
                    {"seconds_a_step": seconds, "held_bytes": held_bytes(device), "failed": failed}
                )
                print(rows[-1], flush=True)
                del encoder, features
                if device.type == "mps":
                    torch.mps.empty_cache()
    arguments.out.mkdir(parents=True, exist_ok=True)
    write_rows(arguments.out / f"cost-{arguments.device}.csv", list(rows[0]), rows)


def cost_row(device: str, resolution: str, length: int, tokens: int, batch: int) -> dict[str, Any]:
    return {
        "device": device,
        "resolution": resolution,
        "length": length,
        "tokens": tokens,
        "batch": batch,
        "seconds_a_step": math.nan,
        "held_bytes": 0,
        "failed": "",
    }


def synchronised(device: Any) -> None:  # pragma: no cover - device
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def held_bytes(device: Any) -> int:  # pragma: no cover - device
    import torch

    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return 0


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every arm's per-month errors on the fitting months, and the descriptions beside."""
    from scripts.taught_fine_tuning_pilot import bench_of
    from scripts.telemetry_context_report import REFIT, Reader

    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    bench = bench_of(arguments)
    reader = Reader(bench, arguments.context)
    fit, _, _ = reader.sides(REFIT, "")
    windows = [reader.windows[i] for i in fit]
    months = [str(w.window.unit).rpartition("/")[2] for w in windows]
    order = sorted(set(months))
    if arguments.months:
        order = order[: arguments.months]
    kept_months = set(order)
    keep = np.asarray([m in kept_months for m in months], dtype=np.bool_)
    fit = [i for i, k in zip(fit, keep, strict=True) if k]
    months = [m for m, k in zip(months, keep, strict=True) if k]
    cut = order[int(len(order) * TEACH_SHARE)]
    inner_cut = order[int(len(order) * TEACH_SHARE * INNER_SHARE)]
    teach = np.asarray([m < cut for m in months], dtype=np.bool_)
    inner = np.asarray([m < inner_cut for m in months], dtype=np.bool_)
    held_out = teach & ~inner
    published = reader.cells
    before = published.previous(fit, max(LENGTHS))
    channels = range(len(published.rows)) if not arguments.channels else arguments.channels
    hours_all = np.concatenate(
        [
            hour_summaries(np.asarray(published.cells[start : start + 4096], dtype=np.float64))
            for start in range(0, len(published.cells), 4096)
        ]
    )
    # Every channel's last hour before each window, as its departure from its own median.
    away = np.concatenate(
        [
            departures(
                gathered(hours_all, before[start : start + 4096, -1]),
                median_of_hours(
                    gathered(hours_all, before[start : start + 4096, -BASELINE_HOURS:])
                )[0],
            )
            for start in range(0, len(fit), 4096)
        ]
    )
    write_rows(
        out / "windows.csv",
        ("month", "windows"),
        [{"month": m, "windows": months.count(m)} for m in order],
    )
    error_rows: list[dict[str, Any]] = []
    penalty_rows: list[dict[str, Any]] = []
    read_channels = 0
    monthly_rows: list[dict[str, Any]] = []
    acf_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    fit_array = np.asarray(fit, dtype=np.int64)
    for channel in channels:
        own = np.asarray(published.cells[:, channel], dtype=np.float64)
        hours = hours_all[:, channel]
        target_hours = hours[fit_array]
        present = target_hours[:, PRESENCE] > 0.0
        # A channel the mission never fills, the other mission's among the corpus's, is skipped.
        if present[teach].sum() < 100:
            continue
        read_channels += 1
        hours = clipped_hours(hours, target_hours[teach & present])
        target_hours = hours[fit_array]
        targets = target_hours[:, list(TARGETS)]
        median, _ = median_of_hours(gathered(hours, before[:, -BASELINE_HOURS:]))
        barred = present & ~np.isnan(median).any(axis=1)
        rows = barred.copy()
        for length in LENGTHS:
            _, held = median_of_hours(gathered(hours, before[:, max(LENGTHS) - length :]))
            rows &= enough_history(held, length)
        median = np.where(np.isnan(median), 0.0, median)
        # The forecast asks what the history adds to the median: the hidden hour's departure.
        departure = targets - median
        scale = floored_scale(np.abs(departure)[teach & rows].mean(axis=0), targets[teach & rows])
        level = median[:, [TARGETS.index(MEAN), TARGETS.index(LOG_COUNT)]]
        others = np.delete(away, channel, axis=1).reshape(len(fit), -1)
        fitted = teach & barred
        inner_rows, held_rows = inner & barred, held_out & barred
        sides = {TEACH: teach & rows, READ: rows & ~teach}
        for resolution in RESOLUTIONS:
            for length in LENGTHS:
                arm = Arm(resolution, length, OWN)
                history = level_free(
                    history_cells(own, before, arm).reshape(len(fit), arm.history_bins, -1), level
                ).reshape(len(fit), -1)
                scaled, kept = standardised(
                    clipped(np.concatenate((history, others), axis=1), fitted), fitted
                )
                inputs = np.concatenate((np.ones((len(fit), 1)), scaled), axis=1)
                places = np.flatnonzero(kept)
                columns = {
                    OWN: np.flatnonzero(np.concatenate(([True], places < history.shape[1]))),
                    CROSS: np.arange(inputs.shape[1]),
                }
                inner_moments = Moments.of(inputs[inner_rows], departure[inner_rows])
                held_moments = Moments.of(inputs[held_rows], departure[held_rows])
                for variant in VARIANTS:
                    penalty = chosen_penalty(
                        inner_moments, inputs[held_rows], departure[held_rows], columns[variant]
                    )
                    penalty_rows.extend(
                        {
                            "resolution": resolution,
                            "length": length,
                            "variant": variant,
                            "channel": int(published.rows[channel]),
                            "target": name,
                            "share": float(share),
                        }
                        for name, share in zip(TARGET_NAMES, penalty, strict=True)
                    )
                    for side, side_rows in sides.items():
                        forecast = median.copy()
                        forecast[side_rows] += forecast_of(
                            inner_moments + held_moments,
                            inputs[side_rows],
                            columns[variant],
                            penalty,
                        )
                        each = month_errors(forecast, median, targets, scale, side_rows, months)
                        error_rows.extend(
                            {
                                "side": side,
                                "resolution": resolution,
                                "length": length,
                                "variant": variant,
                                "month": month,
                                "forecast": each.forecast[month],
                                "median": each.median[month],
                                "cells": each.cells[month],
                                "channel": int(published.rows[channel]),
                            }
                            for month in each.median
                        )
        for month, value in monthly_median_errors(median, targets, scale, rows, months, order):
            monthly_rows.append(
                {
                    "channel": int(published.rows[channel]),
                    "month": month,
                    "median_error": "" if math.isnan(value) else value,
                }
            )
        ends = np.rint([reader.windows[i].window.ends_at for i in fit]).astype(np.int64)
        series = np.full(int(ends.max() - ends.min()) + 1, np.nan)
        series[ends - ends.min()] = np.where(present, targets[:, TARGETS.index(MEAN)], np.nan)
        for lag, value in enumerate(autocorrelation(series, LAGS), start=1):
            acf_rows.append({"channel": int(published.rows[channel]), "lag": lag, "acf": value})
        print(
            f"channel {channel + 1} of {len(published.rows)} done at "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    write_rows(out / "errors.csv", (*ERROR_COLUMNS, "channel"), error_rows)
    write_rows(out / "monthly.csv", ("channel", "month", "median_error"), monthly_rows)
    write_rows(out / "acf.csv", ("channel", "lag", "acf"), acf_rows)
    write_rows(out / "penalties.csv", PENALTY_COLUMNS, penalty_rows)
    conditions = {
        "fitting_windows": len(fit),
        "months": len(order),
        "teach_before": cut,
        "inner_before": inner_cut,
        "channels": read_channels,
        "seconds": round(time.perf_counter() - started),
    }
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="the forecasts' errors on the fitting months")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--task-id", required=True)
    measuring.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    measuring.add_argument("--weights", nargs=3, action="append", required=True)
    measuring.add_argument("--context", type=Path, required=True, help="the context report's run")
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--device", default="cpu")
    measuring.add_argument("--months", type=int, default=0, help="the first months only (trial)")
    measuring.add_argument("--channels", type=int, nargs="*", default=[], help="these only (trial)")
    costing = commands.add_parser("cost", help="forward and backward of each layout, timed")
    costing.add_argument("--experiment", type=Path, required=True)
    costing.add_argument("--device", required=True)
    costing.add_argument("--channels", type=int, default=48)
    costing.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    costing.add_argument("--repeats", type=int, default=3)
    costing.add_argument("--budget-gb", type=float, default=6.0)
    costing.add_argument("--out", type=Path, required=True)
    reading = commands.add_parser("read", help="skills, the chosen layout and the block")
    reading.add_argument("--dir", type=Path, required=True)
    pooling = commands.add_parser("pool", help="the rule over the union of the missions' runs")
    pooling.add_argument("--dirs", type=Path, nargs="+", required=True)
    pooling.add_argument("--names", nargs="+", required=True)
    pooling.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    {"measure": measure, "cost": cost, "read": read, "pool": pool}[arguments.command](arguments)


if __name__ == "__main__":  # pragma: no cover
    main()
