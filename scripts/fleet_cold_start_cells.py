r"""The cells of a wind farm's turbines, each read as a new system from its starts.

A turbine is read as if it had just been switched on: from starts every 35 days through 2020 to
2022, the years the farm's publication held out, from an offset drawn by seed, its hours before
the start are unknown, while the farm's other turbines keep every hour they read. On day N
after the start, drawn hours of the days until the next day read are forecast for each of the
turbine's seventeen component temperatures that read in the hour and in half the 47 before it,
from the 47 hours before, by answers that could have been given then. Every turbine of the
farm is read as new from each start in turn, on the same drawn hours, so a start's turbines
share their weather and form one block of the reading.

- With no learning: the channel's last hour, the median of its 47 hours and the hour one period
  back, as for the satellites (``cold_start_cells.py``). Local forecasters fitted on the same
  hours run in their own environment from the context file written here.
- Learnt from the turbine's own hours since the start, fitted once on day N: a ridge per channel
  over its 47 hours, and one that also reads the window's other channels' latest hour, as for
  the satellites.
- Learnt from the farm's other turbines, before the start and nothing of the new one: the same
  two ridges fitted on every hour each other turbine read before the start, the penalty chosen
  by cross-fitting the calendar months before and after the one in which their rows reach half.
- Beside, and no answer of the bar: the farm's window ridge with the turbine's own power in the
  forecast hour as well as its latest hour before, so it reads the hour's change of load: about
  the most a linear answer that knew the hour's wind could give.
- Of normal behaviour, where the hour's operating conditions are known, as a turbine measures
  them: the cell's own regression over its 47 hours of each hour on the hour before and the
  hour's conditions, which reads what the published forecaster reads; a ridge over the
  channel's 47 hours, the power's last day and six signals of the
  forecast hour — the wind, the power, the rotor's speed, a blade's angle and the two ambient
  temperatures — fitted on the turbine's own hours since the start and, apart, on the farm's
  other turbines before it; beside, the farm's one with the pitch motors' currents of the hour
  too. None is answered where a reading it needs is missing.

A window holds the seventeen temperatures scored and four signals read as context — the wind,
the power and the two ambient temperatures — so the window's ridge and the published forecaster
read the same channels. The farm's boosted trees, of either form, are fitted in a process of
their own (``fleet_tree_answers.py``). The days run to the next day read: on starts whose year
of days lies within the held-out years, 2, 7, 14, 30, 60, 90, 180 and 365, the last for 30
days; on the others 2, 7 and 14, the last to day 30. Starts of one farm lie 35 days apart, so
no two starts' first months share a day.

    uv run scripts/fleet_cold_start_cells.py --farm kelmarsh --hourly DIR --windows 8 \\
        [--seed 1] [--starts 8] [--start-dates 2021-03-04 ...] --out DIR

``--read-from`` and ``--read-until`` name other years for a trial, on which the same rules
then hold; no cell outside them is written. ``--hourly`` holds ``telemetry_hourly_means.py``'s
``hourly-<turbine>.npz`` of every turbine of the farm. Writes ``starts-<farm>.csv`` (each start,
its hour and whether its year is read), and per day N ``context-<farm>-N<days>.csv`` (47 hours
per cell) and ``answers-<farm>-N<days>.csv`` (the target and the answers here); a cell's axis
names its turbine and start, ``<turbine>@<date>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cold_start_cells import (  # noqa: E402
    CONTEXT,
    LEAST,
    NO_LEARNING,
    PENALTIES,
    ROWS_PER_FEATURE,
    Ridge,
    no_learning,
    shown,
)
from scripts.fleet_hours import (  # noqa: E402
    CONDITIONS,
    CURRENTS,
    HOURS_PER_DAY,
    ORIGIN,
    POWER,
    READ_UNTIL,
    TARGETS,
    WINDOW,
    Turbine,
    axis_of,
    hour_of,
    loaded_farm,
)
from scripts.local_forecaster_answers import filled  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Mask = NDArray[np.bool_]

READ_FROM = date(2020, 1, 1)
VERDICT_DAYS = (2, 7, 14)
CURVE_DAYS = (2, 7, 14, 30, 60, 90, 180, 365)
FIRST_MONTH = 30
SPACING_DAYS = 35
OWN = ("ridge", "window_ridge")
FLEET = ("fleet_ridge", "fleet_window_ridge")
ORACLE = "oracle_power_ridge"
OWN_NBM = "nbm_ridge"
FLEET_NBM = "fleet_nbm_ridge"
FLEET_NBM_CURRENTS = "fleet_nbm_ridge_currents"
LOCAL_NBM = "local_nbm_arx"
# The local normal-behaviour answer's penalty, on standardised features: the best of 0.01 to 100
# by its own error over the last hour's on the trial's first months, both farms together.
ARX_PENALTY = 0.1
# How many of the hours before the power's history the normal-behaviour answers read: a day,
# the thermal load the temperatures carry.
LOAD_HOURS = 24


def scorable(values: Floats) -> Mask:
    """``[hours, channels]``: a channel read in the hour and in ``LEAST`` of the 47 before."""
    read = np.isfinite(values)
    counts = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(read, axis=0)])
    out = np.zeros_like(read)
    hours = np.arange(CONTEXT, len(values))
    out[CONTEXT:] = read[CONTEXT:] & (counts[hours] - counts[hours - CONTEXT] >= LEAST)
    return out


def latest(values: Floats) -> Floats:
    """``[hours, channels]``: each channel's latest reading in the 47 hours before, nought if none.

    Nought is the median the robust score centres on, as the satellites' window ridge took it.
    """
    hours, channels = values.shape
    read = np.isfinite(values)
    last = np.where(read, np.arange(hours)[:, None], -1)
    np.maximum.accumulate(last, axis=0, out=last)
    before = np.full((hours, channels), -1, dtype=np.int64)
    before[1:] = last[:-1]
    recent = (before >= 0) & (np.arange(hours)[:, None] - before <= CONTEXT)
    taken = values[np.maximum(before, 0), np.arange(channels)]
    return np.where(recent, taken, 0.0)


def filled_rows(windows: Floats) -> Floats:
    """Each row with a missing hour carrying the hour before, the first reading backwards.

    The same as ``local_forecaster_answers.filled`` row by row: a row is filled from its own
    hours, never from the hours before it.

    Raises:
        ValueError: If a row holds no reading.
    """
    read = np.isfinite(windows)
    if not read.any(axis=1).all():
        raise ValueError("a context with no hour read cannot be filled")
    width = windows.shape[1]
    last = np.where(read, np.arange(width), 0)
    np.maximum.accumulate(last, axis=1, out=last)
    first = read.argmax(axis=1)
    last = np.where(np.arange(width) < first[:, None], first[:, None], last)
    return np.take_along_axis(windows, last, axis=1)


def filled_or_missing(windows: Floats) -> Floats:
    """``filled_rows`` where a row read in some hour, and missing where it read in none."""
    read = np.isfinite(windows).any(axis=1)
    out = np.full(windows.shape, np.nan)
    if read.any():
        out[read] = filled_rows(windows[read])
    return out


@dataclass(frozen=True)
class Rows:
    """What a ridge reads for a channel's hour, beside the channel's own 47 hours filled.

    Attributes:
        name: The answer the ridge writes.
        window: The window's other channels' latest hour before, nought where none read.
        power: The turbine's power in the forecast hour itself: the oracle's.
        conditions: The channels read in the forecast hour: the normal-behaviour answers'.
        load: The power's last day of hours before, filled from its own hours.
    """

    name: str
    window: bool = False
    power: bool = False
    conditions: tuple[int, ...] = ()
    load: bool = False

    def of(self, values: Floats, recent: Floats, hours: Ints, channel: int) -> Floats:
        """``[hours, features]``; a row holding a missing reading is neither fitted nor answered.

        ``recent`` is ``latest(values)``.
        """
        parts = [filled_rows(sliding_window_view(values[:, channel], CONTEXT)[hours - CONTEXT])]
        if self.window:
            parts.append(np.delete(recent[hours, : len(WINDOW)], channel, axis=1))
        if self.power:
            parts.append(values[hours, POWER][:, None])
        if self.conditions:
            parts.append(values[hours][:, list(self.conditions)])
        if self.load:
            history = sliding_window_view(values[:, POWER], CONTEXT)[hours - CONTEXT]
            parts.append(filled_or_missing(history)[:, -LOAD_HOURS:])
        return np.concatenate(parts, axis=1)


OWN_ROWS = (
    Rows(OWN[0]),
    Rows(OWN[1], window=True),
    Rows(OWN_NBM, conditions=CONDITIONS, load=True),
)
FLEET_ROWS = (
    Rows(FLEET[0]),
    Rows(FLEET[1], window=True),
    Rows(ORACLE, window=True, power=True),
    Rows(FLEET_NBM, conditions=CONDITIONS, load=True),
    Rows(FLEET_NBM_CURRENTS, conditions=(*CONDITIONS, *CURRENTS), load=True),
)
ANSWERS = (
    *NO_LEARNING,
    LOCAL_NBM,
    *(r.name for r in OWN_ROWS),
    *(r.name for r in FLEET_ROWS),
)
ANSWER_COLUMNS = (
    "window",
    "axis",
    "channel",
    "lag",
    "target_mean",
    *(f"mean_{name}" for name in ANSWERS),
)


def local_arx(history: Floats, conditions: Floats, now: Floats, penalty: float) -> float | None:
    """The cell's own normal-behaviour answer: a ridge over its 47 hours alone.

    Each hour of the context read, whose hour before was read too, is regressed on that hour
    and on the hour's operating conditions; the forecast hour is answered from the last hour,
    carried from the last one read where it is missing, and its own conditions. The answer
    reads what the published forecaster of normal behaviour reads on 47 hours, and nothing of
    the turbine's other hours or of the farm. None where the hour's conditions are missing or
    fewer than twice as many transitions as features read.
    """
    if not np.isfinite(now).all():
        return None
    level = filled(history)
    rows = np.c_[level[:-1], conditions[1:]]
    kept = np.isfinite(rows).all(axis=1) & np.isfinite(history[1:]) & np.isfinite(history[:-1])
    if kept.sum() < ROWS_PER_FEATURE * rows.shape[1]:
        return None
    rows, targets = rows[kept], history[1:][kept]
    centre, scale = rows.mean(axis=0), rows.std(axis=0)
    scale[scale == 0] = 1.0
    z = (rows - centre) / scale
    weights = np.linalg.solve(
        z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (targets - targets.mean())
    )
    return float(((np.r_[level[-1], now] - centre) / scale) @ weights + targets.mean())


def answers_of(ridge: Ridge, rows: Floats) -> Floats:
    """``Ridge.answer`` for every row."""
    out: Floats = np.clip(rows, ridge.low, ridge.high) @ ridge.weights + ridge.intercept
    return out


@dataclass(frozen=True)
class Moments:
    """A ridge's rows summed: enough to fit it and score it on them, without the rows.

    Sums over rows are added for several turbines; the least and largest feature too.
    """

    count: int
    x: Floats
    y: float
    xx: Floats
    xy: Floats
    yy: float
    low: Floats
    high: Floats

    @classmethod
    def of(cls, rows: Floats, targets: Floats) -> Moments:
        width = rows.shape[1]
        if not len(rows):
            empty = np.full(width, np.nan)
            return cls(
                0,
                np.zeros(width),
                0.0,
                np.zeros((width, width)),
                np.zeros(width),
                0.0,
                empty,
                empty,
            )
        return cls(
            len(rows),
            rows.sum(axis=0),
            float(targets.sum()),
            rows.T @ rows,
            rows.T @ targets,
            float(targets @ targets),
            rows.min(axis=0),
            rows.max(axis=0),
        )

    @classmethod
    def summed(cls, parts: Sequence[Moments]) -> Moments:
        return cls(
            sum(p.count for p in parts),
            np.sum([p.x for p in parts], axis=0),
            sum(p.y for p in parts),
            np.sum([p.xx for p in parts], axis=0),
            np.sum([p.xy for p in parts], axis=0),
            sum(p.yy for p in parts),
            np.fmin.reduce([p.low for p in parts]),
            np.fmax.reduce([p.high for p in parts]),
        )

    def solved(self, penalty: float) -> tuple[Floats, float]:
        """``Ridge.solved`` on the rows these moments sum."""
        centre, level = self.x / self.count, self.y / self.count
        gram = self.xx - self.count * np.outer(centre, centre)
        cross = self.xy - self.count * centre * level
        weights = np.linalg.solve(gram + penalty * np.eye(len(centre)), cross)
        return weights, float(level - centre @ weights)

    def missed(self, weights: Floats, intercept: float) -> float:
        """The squared error summed over the rows, of an answer fitted elsewhere, unclipped."""
        return float(
            self.yy
            - 2.0 * weights @ self.xy
            - 2.0 * intercept * self.y
            + weights @ self.xx @ weights
            + 2.0 * intercept * weights @ self.x
            + self.count * intercept**2
        )


def fitted_from(first: Moments, second: Moments) -> Ridge | None:
    """A ridge on the rows of both halves, or None with too few rows or a half empty.

    The penalty is the one whose fit on each half answers the other half best.
    """
    whole = Moments.summed([first, second])
    if whole.count < ROWS_PER_FEATURE * len(whole.x) or not first.count or not second.count:
        return None

    def missed(penalty: float) -> float:
        return second.missed(*first.solved(penalty)) + first.missed(*second.solved(penalty))

    penalty = min(PENALTIES, key=missed)
    weights, intercept = whole.solved(penalty)
    return Ridge(weights, intercept, whole.low, whole.high, penalty)


@dataclass(frozen=True)
class Years:
    """The years a reading reads, ``until`` exclusive: the held-out years, or a trial's."""

    first: date
    until: date


READ = Years(READ_FROM, READ_UNTIL)


def systematic_starts(
    seed: Sequence[int], years: Years = READ, count: int | None = None
) -> list[date]:
    """Starts every 35 days through the years read, from an offset drawn by ``seed``.

    The latest is the last whose first month is read, so every start's first month lies apart
    from the others' and the starts cover the seasons evenly. ``seed`` holds the farm's name as
    well, so two farms' offsets are drawn apart. With ``count``, that many of them, as evenly
    spread over the series as they go, as a trial reads.

    Raises:
        ValueError: If ``count`` exceeds the starts there are.
    """
    last = years.until - timedelta(days=FIRST_MONTH)
    offset = int(np.random.default_rng(seed).integers(0, SPACING_DAYS))
    starts = []
    day = years.first + timedelta(days=offset)
    while day <= last:
        starts.append(day)
        day += timedelta(days=SPACING_DAYS)
    if count is None:
        return starts
    if count > len(starts):
        raise ValueError(f"only {len(starts)} starts fit in the years read, not {count}")
    picks = np.unique(np.round(np.linspace(0, len(starts) - 1, count)).astype(int))
    return [starts[k] for k in picks.tolist()]


def farm_seed(seed: int, farm: str) -> tuple[int, int]:
    """The draw's seed with the farm's name, so each farm's starts are drawn on their own."""
    return seed, zlib.crc32(farm.encode())


def days_of(start: date, years: Years = READ) -> tuple[tuple[int, ...], dict[int, int]]:
    """A start's days read and each day's last day, exclusive.

    A day runs to the next day read; the curve's last day for 30 days, the first month's to
    day 30. The curve is read only where its last day lies within the years read.
    """
    curve = start + timedelta(days=CURVE_DAYS[-1] + FIRST_MONTH) <= years.until
    days = CURVE_DAYS if curve else VERDICT_DAYS
    ends = [*days[1:], days[-1] + FIRST_MONTH if curve else FIRST_MONTH]
    return days, dict(zip(days, ends, strict=True))


def drawn_hours(scored: Sequence[Mask], begin: int, end: int, count: int, seed: int) -> Ints:
    """The hours of ``[begin, end)`` in which a turbine scores a target, sorted.

    All of them, or ``count`` drawn by ``seed``; none if no hour does.
    """
    candidates = np.arange(begin, end)
    candidates = candidates[
        np.any([s[begin:end, : len(TARGETS)].any(axis=1) for s in scored], axis=0)
    ]
    if candidates.size <= count:
        return candidates
    return np.sort(np.random.default_rng(seed).choice(candidates, size=count, replace=False))


def month_starts(before: int) -> Ints:
    """The hour each calendar month from the origin starts, up to ``before``."""
    out, day = [], ORIGIN
    while hour_of(day) < before:
        out.append(hour_of(day))
        day = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return np.asarray(out, dtype=np.int64)


def halved(months: Sequence[Moments]) -> tuple[Moments, Moments] | None:
    """The months up to the one in which the rows reach half, and the months after it.

    None where no row is summed or half the rows are reached only in the last month; the
    second half may then be empty, which ``fitted_from`` refuses.
    """
    counts = np.cumsum([m.count for m in months])
    if not counts.size or not counts[-1]:
        return None
    middle = int(np.searchsorted(counts, counts[-1] / 2)) + 1
    if middle >= len(months):
        return None
    return Moments.summed(months[:middle]), Moments.summed(months[middle:])


@dataclass(frozen=True)
class FleetFits:
    """The farm's ridges for each turbine read as new from a start, fitted on the others.

    Each turbine's rows before the start are summed by calendar month; a new turbine's ridge
    sums the other turbines' months and cross-fits the months before and after the one in which
    their rows reach half, so nothing of the new turbine, not even which hours it read, enters.
    """

    ridges: Mapping[tuple[str, int, str], Ridge | None]

    @classmethod
    def before(cls, turbines: Sequence[Turbine], start: int) -> FleetFits:
        out: dict[tuple[str, int, str], Ridge | None] = {}
        scored = {t.name: scorable(t.values[:start]) for t in turbines}
        recent = {t.name: latest(t.values) for t in turbines}
        months = month_starts(start)
        for channel in range(len(TARGETS)):
            for kind in FLEET_ROWS:
                name = kind.name
                monthly: dict[str, list[Moments]] = {}
                for turbine in turbines:
                    hours = np.flatnonzero(scored[turbine.name][:, channel])
                    rows = kind.of(turbine.values, recent[turbine.name], hours, channel)
                    kept = np.isfinite(rows).all(axis=1)
                    hours, rows = hours[kept], rows[kept]
                    targets = turbine.values[hours, channel]
                    edges = np.searchsorted(hours, [*months.tolist(), start])
                    monthly[turbine.name] = [
                        Moments.of(rows[lo:hi], targets[lo:hi])
                        for lo, hi in pairwise(edges.tolist())
                    ]
                for new in turbines:
                    others = [m for t, m in monthly.items() if t != new.name]
                    halves = halved([Moments.summed(parts) for parts in zip(*others, strict=True)])
                    out[(new.name, channel, name)] = (
                        None if halves is None else fitted_from(*halves)
                    )
        return cls(out)

    def fitted(self, new: str, channel: int, name: str) -> Ridge | None:
        return self.ridges[(new, channel, name)]


@dataclass(frozen=True)
class Cell:
    window: int
    axis: str
    channel: int
    target: float
    history: Floats
    lag: int | None
    answers: Mapping[str, float | None]


def own_ridges(
    new: Turbine, start: int, before: int, scored: Mask, recent: Floats
) -> dict[str, dict[int, Ridge]]:
    """The ridges fitted on the new turbine's hours from its start to ``before``.

    ``scored`` and ``recent`` are ``scorable`` and ``latest`` of its values.
    """
    out: dict[str, dict[int, Ridge]] = {kind.name: {} for kind in OWN_ROWS}
    for channel in range(len(TARGETS)):
        hours = np.flatnonzero(scored[start:before, channel]) + start
        if not hours.size:
            continue
        for kind in OWN_ROWS:
            rows = kind.of(new.values, recent, hours, channel)
            kept = np.isfinite(rows).all(axis=1)
            ridge = Ridge.fitted(rows[kept], new.values[hours[kept], channel])
            if ridge is not None:
                out[kind.name][channel] = ridge
    return out


def cells_of(
    new: Turbine,
    axis: str,
    hours: Ints,
    own: Mapping[str, Mapping[int, Ridge]],
    fleet: FleetFits,
    scored: Mask,
    recent: Floats,
) -> list[Cell]:
    """The new turbine's cells in ``hours``, a channel's answers given together.

    A ridge answers a cell only where every reading it needs is there; ``scored`` and
    ``recent`` are ``scorable`` and ``latest`` of the turbine's values.
    """
    cells: list[Cell] = []
    for channel in range(len(TARGETS)):
        at = hours[scored[hours, channel]]
        if not at.size:
            continue
        given: dict[str, Floats] = {}
        for kind, ridge in (
            *((kind, own[kind.name].get(channel)) for kind in OWN_ROWS),
            *((kind, fleet.fitted(new.name, channel, kind.name)) for kind in FLEET_ROWS),
        ):
            if ridge is None:
                continue
            rows = kind.of(new.values, recent, at, channel)
            kept = np.isfinite(rows).all(axis=1)
            answered = np.full(len(at), np.nan)
            if kept.any():
                answered[kept] = answers_of(ridge, rows[kept])
            given[kind.name] = answered
        for k, hour in enumerate(at.tolist()):
            history = new.values[hour - CONTEXT : hour, channel]
            plain, lag = no_learning(history)
            answers: dict[str, float | None] = dict(plain)
            answers[LOCAL_NBM] = local_arx(
                history,
                new.values[hour - CONTEXT : hour][:, list(CONDITIONS)],
                new.values[hour, list(CONDITIONS)],
                ARX_PENALTY,
            )
            for name in (kind.name for kind in (*OWN_ROWS, *FLEET_ROWS)):
                value = given[name][k] if name in given else np.nan
                answers[name] = float(value) if np.isfinite(value) else None
            cells.append(
                Cell(hour, axis, channel, float(new.values[hour, channel]), history, lag, answers)
            )
    return cells


def write_cells(directory: Path, label: str, cells: Sequence[Cell]) -> None:
    with (directory / f"context-{label}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("window", "axis", "channel", *(f"h{h:02d}" for h in range(1, CONTEXT + 1)))
        )
        for cell in cells:
            writer.writerow(
                (cell.window, cell.axis, cell.channel, *(shown(h) for h in cell.history))
            )
    with (directory / f"answers-{label}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANSWER_COLUMNS)
        for cell in cells:
            writer.writerow(
                (
                    cell.window,
                    cell.axis,
                    cell.channel,
                    "" if cell.lag is None else cell.lag,
                    f"{cell.target:.6g}",
                    *(shown(cell.answers[name]) for name in ANSWERS),
                )
            )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--farm", required=True)
    parser.add_argument("--hourly", type=Path, required=True, help="telemetry_hourly_means's")
    parser.add_argument("--starts", type=int, help="fewer starts, spread over the series")
    parser.add_argument(
        "--start-dates", type=date.fromisoformat, nargs="+", help="starts named instead of drawn"
    )
    parser.add_argument("--windows", type=int, required=True, help="hours per start and day")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--read-from", type=date.fromisoformat, default=READ_FROM)
    parser.add_argument("--read-until", type=date.fromisoformat, default=READ_UNTIL)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    years = Years(arguments.read_from, arguments.read_until)

    turbines = loaded_farm(arguments.hourly)
    names = [t.name for t in turbines]
    starts = (
        sorted(arguments.start_dates)
        if arguments.start_dates
        else systematic_starts(farm_seed(arguments.seed, arguments.farm), years, arguments.starts)
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    by_day: dict[int, list[Cell]] = {}
    with (arguments.out / f"starts-{arguments.farm}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("start", "hour", "curve"))
        for start in starts:
            begin = hour_of(start)
            days, ends = days_of(start, years)
            writer.writerow((start.isoformat(), begin, int(len(days) > len(VERDICT_DAYS))))
            fleet = FleetFits.before(turbines, begin)
            news = [t.before(begin) for t in turbines]
            scored = [scorable(t.values) for t in news]
            recent = [latest(t.values) for t in news]
            for day in days:
                hours = drawn_hours(
                    scored,
                    begin + day * HOURS_PER_DAY,
                    begin + ends[day] * HOURS_PER_DAY,
                    arguments.windows,
                    arguments.seed * 1_000_003 + begin + day,
                )
                for new, mask, last in zip(news, scored, recent, strict=True):
                    own = own_ridges(new, begin, begin + day * HOURS_PER_DAY, mask, last)
                    by_day.setdefault(day, []).extend(
                        cells_of(new, axis_of(new.name, start), hours, own, fleet, mask, last)
                    )
            print(f"{arguments.farm} {start}: days {list(days)}", flush=True)
    outside = [
        c
        for cells in by_day.values()
        for c in cells
        if not hour_of(years.first) <= c.window < hour_of(years.until)
    ]
    if outside:
        raise SystemExit(f"{len(outside)} cells lie outside {years.first} to {years.until}")
    for day, cells in sorted(by_day.items()):
        write_cells(arguments.out, f"{arguments.farm}-N{day}", cells)
        print(f"{arguments.farm}-N{day}: {len(cells)} cells")
    (arguments.out / f"conditions-{arguments.farm}.json").write_text(
        json.dumps(
            {
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(arguments).items()
                    if k != "start_dates"
                },
                "starts_read": [s.isoformat() for s in starts],
                "turbines": names,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
