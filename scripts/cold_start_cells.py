r"""The cells of a satellite read as a new system, N days after its first reading.

A mission is read as if it had just been switched on: on day N of its data, each hour of the
days until the next day read (30 days after the last), or a draw of them where there are more,
is forecast for every channel that read in it, from the channel's 47 hours before, by answers
that could have been given then. An answer that learns from the mission's history learns from
the hours before day N alone, and is used until the next day read, so the days' hours are
disjoint and can be read together.

- With no learning: the channel's last hour, the median of its 47 hours, and the hour one period
  back (the same-phase answer of the forecasting pretext's reading, its period chosen from those
  hours). Local forecasters fitted on the same 47 hours run in their own environment from the
  context file written here (``local_forecaster_answers.py``).
- Learnt from the hours before day N, fitted once on day N: a ridge per channel over its 47
  hours, and one that also reads the other channels' latest hour within those 47, nought (the
  median the robust score centres on) for one that read none of them; the penalty chosen by
  two-fold cross-fitting on the first and second half of those hours from a grid of eleven
  decades, an answer's inputs clipped to the range it was fitted on. A ridge answers once it has
  twice as many rows as features: the one per channel from day 7, the window's, of about twice
  as many features, from day 14.
- For the published forecaster (``chronos2_ladder.py``): a library series per channel that has a
  counterpart on the other mission, a target channel's kind being its subsystem and physical
  unit, drawn from the other mission's fitting side in three ways — a random channel of another
  kind at a random hour, a random channel of the same kind at a random hour, and the series of
  the same kind whose 47 hours ending at some hour are nearest to the target's own 47 hours, each
  z-scored. The chosen hour is where the library series meets the
  forecast hour, so the hour after it stands for the forecast hour, known.

    uv run scripts/cold_start_cells.py --mission ESA-Mission2 --library ESA-Mission1 \\
        --hourly DIR --days 2 7 14 30 --windows 300 --library-end 61368 --out DIR

``--library-end`` is the first hour after the other mission's fitting side. Writes, per day N,
``context-<mission>-N<days>.csv`` (47 hours per cell), ``answers-<mission>-N<days>.csv`` (the
target and the answers here) and ``selections-<mission>-N<days>.csv`` (each library series:
the rung, the library channel and the hour it meets the forecast hour).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.history_pretext_answers_report import same_phase  # noqa: E402
from scripts.local_forecaster_answers import filled  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

# The hours a cell's answers read, as the forecast was read on 2026-11-15 and 2026-11-19.
CONTEXT = 47
# A channel is scored only where half the hours before it read.
LEAST = 24
BLOCK_DAYS = 30
HOURS_PER_DAY = 24
# Two decades wider on either side than the forecast gate's grid: on the trial its edges were
# chosen for most channels of the first mission.
PENALTIES = tuple(10.0**k for k in range(-4, 7))
# A ridge answers only once its rows outnumber its features this many times over.
ROWS_PER_FEATURE = 2
# A guard for a z-scored 47 hours that never moved, which is then all noughts; otherwise the
# shape is matched at whatever scale it moves, as subsequence search z-normalises.
DEVIATION_FLOOR = 1e-6
RUNGS = ("unrelated", "kind", "analog")
NO_LEARNING = ("last_hour", "median_of_history", "same_phase")
LEARNT = ("ridge", "window_ridge")
ANSWER_COLUMNS = (
    "window",
    "axis",
    "channel",
    "lag",
    "target_mean",
    *(f"mean_{name}" for name in (*NO_LEARNING, *LEARNT)),
)
SELECTION_COLUMNS = ("window", "axis", "channel", "rung", "library_channel", "meets", "distance")


@dataclass(frozen=True)
class Mission:
    """A mission's hourly means on a dense axis from its first hour, and its channels' kinds.

    ``values`` is ``[hours, channels]``, NaN where a channel read nothing or the hour was not
    loaded; ``kinds`` each channel's subsystem and physical unit.
    """

    name: str
    channels: tuple[str, ...]
    kinds: tuple[str, ...]
    values: Floats

    @classmethod
    def dense(
        cls,
        name: str,
        channels: Sequence[str],
        kinds: Sequence[str],
        hours: Ints,
        means: Floats,
    ) -> Mission:
        if len(channels) != len(kinds) or means.shape != (len(hours), len(channels)):
            raise ValueError("the channels, kinds and means do not agree")
        values = np.full((int(hours.max()) + 1, len(channels)), np.nan)
        values[hours] = means
        return cls(name, tuple(channels), tuple(kinds), values)

    def scorable(self, hour: int) -> Ints:
        """The channels that read in ``hour`` and in at least ``LEAST`` of the 47 before it."""
        if hour < CONTEXT or hour >= len(self.values):
            return np.zeros(0, dtype=np.int64)
        read = np.isfinite(self.values[hour - CONTEXT : hour]).sum(axis=0)
        return np.flatnonzero(np.isfinite(self.values[hour]) & (read >= LEAST))

    def history(self, hour: int, channel: int, hours: int = CONTEXT) -> Floats:
        """The channel's ``hours`` before ``hour``, NaN before the first."""
        start = hour - hours
        seen = self.values[max(start, 0) : hour, channel]
        return np.concatenate([np.full(max(-start, 0), np.nan), seen])


def kind_of(subsystem: str, unit: str) -> str:
    return f"{subsystem}/{unit or 'none'}"


def drawn_hours(mission: Mission, start: int, end: int, count: int, seed: int) -> Ints:
    """The hours of ``[start, end)`` that score a channel, or ``count`` drawn by ``seed``, sorted.

    Raises:
        ValueError: If no hour scores one.
    """
    candidates = np.asarray(
        [h for h in range(start, min(end, len(mission.values))) if mission.scorable(h).size],
        dtype=np.int64,
    )
    if not candidates.size:
        raise ValueError(f"{mission.name}: no hour of [{start}, {end}) scores a channel")
    if candidates.size <= count:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=count, replace=False))


def block_ends(days: Sequence[int]) -> dict[int, int]:
    """Each day's last day read, exclusive: the next day, and 30 days after the last.

    Raises:
        ValueError: If the days are not distinct and rising.
    """
    if list(days) != sorted(set(days)):
        raise ValueError(f"the days must be distinct and rising, got {list(days)}")
    nexts = [*days[1:], days[-1] + BLOCK_DAYS]
    return dict(zip(days, nexts, strict=True))


def no_learning(history: Floats) -> tuple[dict[str, float], int | None]:
    """The answers that learn nothing, and the period the same-phase answer went back."""
    read = np.isfinite(history)
    cycled, lag = same_phase(np.where(read, history, 0.0), read)
    return {
        "last_hour": float(history[np.flatnonzero(read)[-1]]),
        "median_of_history": float(np.median(history[read])),
        "same_phase": float(cycled),
    }, lag


@dataclass(frozen=True)
class Ridge:
    """A ridge fitted on rows of the hours before day N; answers clipped to the fitted range."""

    weights: Floats
    intercept: float
    low: Floats
    high: Floats
    penalty: float

    @staticmethod
    def solved(features: Floats, targets: Floats, penalty: float) -> tuple[Floats, float]:
        centre, level = features.mean(axis=0), targets.mean()
        x = features - centre
        gram = x.T @ x + penalty * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x.T @ (targets - level))
        return weights, float(level - centre @ weights)

    @classmethod
    def fitted(cls, features: Floats, targets: Floats) -> Ridge | None:
        """The ridge on these rows, oldest first, or None with too few of them.

        The penalty is the one whose fit on each half answers the other half best.
        """
        rows, width = features.shape
        if rows < ROWS_PER_FEATURE * width:
            return None
        half = rows // 2

        def missed(penalty: float) -> float:
            total = 0.0
            for fit, held in (
                (slice(0, half), slice(half, rows)),
                (slice(half, rows), slice(0, half)),
            ):
                weights, intercept = cls.solved(features[fit], targets[fit], penalty)
                total += float(((features[held] @ weights + intercept - targets[held]) ** 2).sum())
            return total

        penalty = min(PENALTIES, key=missed)
        weights, intercept = cls.solved(features, targets, penalty)
        return cls(weights, intercept, features.min(axis=0), features.max(axis=0), penalty)

    def answer(self, features: Floats) -> float:
        return float(np.clip(features, self.low, self.high) @ self.weights + self.intercept)


def ridge_features(mission: Mission, hour: int, channel: int, window: bool) -> Floats:
    """A row: the channel's 47 hours filled; for the window's ridge the others' latest hour."""
    own = filled(mission.history(hour, channel))
    if not window:
        return own
    context = mission.values[hour - CONTEXT : hour]
    read = np.isfinite(context)
    latest = np.where(read.any(axis=0), CONTEXT - 1 - np.argmax(read[::-1], axis=0), 0)
    last = np.where(read.any(axis=0), context[latest, np.arange(context.shape[1])], 0.0)
    return np.concatenate([own, np.delete(last, channel)])


def fitted_ridges(mission: Mission, before: int, window: bool) -> dict[int, Ridge]:
    """A ridge per channel on every hour before ``before`` the channel is scored in."""
    rows: dict[int, list[tuple[Floats, float]]] = {}
    for hour in range(CONTEXT, before):
        for channel in mission.scorable(hour).tolist():
            rows.setdefault(channel, []).append(
                (
                    ridge_features(mission, hour, channel, window),
                    float(mission.values[hour, channel]),
                )
            )
    ridges = {}
    for channel, pairs in rows.items():
        ridge = Ridge.fitted(np.stack([f for f, _ in pairs]), np.asarray([t for _, t in pairs]))
        if ridge is not None:
            ridges[channel] = ridge
    return ridges


def zscored(rows: Floats) -> Floats:
    """Each row less its mean over its deviation; a row that never moved is all noughts."""
    deviation = np.maximum(rows.std(axis=1, keepdims=True), DEVIATION_FLOOR)
    return (rows - rows.mean(axis=1, keepdims=True)) / deviation


@dataclass(frozen=True)
class Library:
    """The other mission's fitting side, by kind: every 47 hours fully read with the hour after.

    ``meets[i]`` is the hour after segment ``i``, the one a forecast hour is met by. A channel's
    segments are z-scored once and kept.
    """

    mission: Mission
    end: int
    _kept: dict[int, tuple[Floats, Ints]] = field(default_factory=dict, repr=False)

    def segments(self, channel: int) -> tuple[Floats, Ints]:
        if channel not in self._kept:
            series = self.mission.values[: self.end, channel]
            windows = sliding_window_view(series[:-1], CONTEXT)
            meets = np.arange(CONTEXT, len(series), dtype=np.int64)
            whole = np.isfinite(windows).all(axis=1) & np.isfinite(series[meets])
            self._kept[channel] = (zscored(windows[whole]), meets[whole])
        return self._kept[channel]

    def channels_of(self, kind: str, same: bool) -> list[int]:
        return [c for c, k in enumerate(self.mission.kinds) if (k == kind) == same]


def nearest(
    queries: Floats, library: Library, channels: Sequence[int]
) -> list[tuple[int, int, float]]:
    """For each z-scored query, the library channel and meeting hour nearest it, and how near.

    Raises:
        ValueError: If no channel of those named holds a whole segment.
    """
    best = np.full(len(queries), np.inf)
    found = np.zeros((len(queries), 2), dtype=np.int64)
    squares = (queries**2).sum(axis=1)
    for channel in channels:
        segments, meets = library.segments(channel)
        if not len(meets):
            continue
        norms = (segments**2).sum(axis=1)
        for start in range(0, len(queries), 256):
            part = slice(start, start + 256)
            distances = squares[part, None] + norms[None, :] - 2.0 * queries[part] @ segments.T
            at = distances.argmin(axis=1)
            closer = distances[np.arange(len(at)), at] < best[part]
            rows = np.flatnonzero(closer) + start
            best[rows] = distances[np.flatnonzero(closer), at[closer]]
            found[rows, 0] = channel
            found[rows, 1] = meets[at[closer]]
    if not np.isfinite(best).all():
        raise ValueError(f"no library channel of {list(channels)} holds a whole segment")
    return [
        (int(c), int(m), float(np.sqrt(max(d, 0.0)))) for (c, m), d in zip(found, best, strict=True)
    ]


def random_segment(
    library: Library, channels: Sequence[int], rng: np.random.Generator
) -> tuple[int, int]:
    """A channel of those named that holds a whole segment, and one of its meeting hours.

    Raises:
        ValueError: If none of them holds one.
    """
    whole = [c for c in channels if len(library.segments(c)[1])]
    if not whole:
        raise ValueError(f"no library channel of {list(channels)} holds a whole segment")
    channel = int(rng.choice(np.asarray(whole)))
    return channel, int(rng.choice(library.segments(channel)[1]))


@dataclass(frozen=True)
class Cell:
    window: int
    channel: int
    target: float
    history: Floats
    lag: int | None
    answers: Mapping[str, float | None]


def cells_of(
    mission: Mission, hours: Ints, ridges: Mapping[str, Mapping[int, Ridge]]
) -> list[Cell]:
    cells = []
    for hour in hours.tolist():
        for channel in mission.scorable(hour).tolist():
            history = mission.history(hour, channel)
            answers: dict[str, float | None]
            plain, lag = no_learning(history)
            answers = dict(plain)
            for name, window in (("ridge", False), ("window_ridge", True)):
                ridge = ridges[name].get(channel)
                answers[name] = (
                    None
                    if ridge is None
                    else ridge.answer(ridge_features(mission, hour, channel, window))
                )
            target = float(mission.values[hour, channel])
            cells.append(Cell(hour, channel, target, history, lag, answers))
    return cells


def selections_of(
    mission: Mission, cells: Sequence[Cell], library: Library, seed: int
) -> list[tuple[int, int, str, int, int, float]]:
    """Each cell with a counterpart: a library series per rung, and the analog's distance."""
    rng = np.random.default_rng(seed)
    kinds = set(library.mission.kinds)
    chosen = [c for c in cells if mission.kinds[c.channel] in kinds]
    out: list[tuple[int, int, str, int, int, float]] = []
    by_kind: dict[str, list[int]] = {}
    for index, cell in enumerate(chosen):
        by_kind.setdefault(mission.kinds[cell.channel], []).append(index)
    analogs: dict[int, tuple[int, int, float]] = {}
    for kind, indices in by_kind.items():
        queries = zscored(np.stack([filled(chosen[i].history) for i in indices]))
        for i, found in zip(
            indices, nearest(queries, library, library.channels_of(kind, True)), strict=True
        ):
            analogs[i] = found
    for index, cell in enumerate(chosen):
        kind = mission.kinds[cell.channel]
        other = random_segment(library, library.channels_of(kind, False), rng)
        same = random_segment(library, library.channels_of(kind, True), rng)
        analog = analogs[index]
        out.append((cell.window, cell.channel, "unrelated", *other, float("nan")))
        out.append((cell.window, cell.channel, "kind", *same, float("nan")))
        out.append((cell.window, cell.channel, "analog", *analog))
    return out


def shown(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.6g}"


def write_cells(directory: Path, label: str, axis: str, cells: Sequence[Cell]) -> None:
    with (directory / f"context-{label}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("window", "axis", "channel", *(f"h{h:02d}" for h in range(1, CONTEXT + 1)))
        )
        for cell in cells:
            writer.writerow((cell.window, axis, cell.channel, *(shown(h) for h in cell.history)))
    with (directory / f"answers-{label}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANSWER_COLUMNS)
        for cell in cells:
            writer.writerow(
                (
                    cell.window,
                    axis,
                    cell.channel,
                    "" if cell.lag is None else cell.lag,
                    f"{cell.target:.6g}",
                    *(shown(cell.answers[name]) for name in (*NO_LEARNING, *LEARNT)),
                )
            )


def write_selections(
    directory: Path, label: str, axis: str, rows: Sequence[tuple[int, int, str, int, int, float]]
) -> None:
    with (directory / f"selections-{label}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SELECTION_COLUMNS)
        for window, channel, rung, library_channel, meets, distance in rows:
            writer.writerow((window, axis, channel, rung, library_channel, meets, shown(distance)))


def loaded(directory: Path, name: str, raw: Path) -> Mission:
    """A mission's hourly means and its channels' kinds from the benchmark's channel table."""
    with np.load(directory / f"hourly-{name}.npz") as arrays:
        channels = [str(c) for c in arrays["channels"]]
        hours, means = arrays["hours"], arrays["means"].astype(np.float64)
    with (raw / name / "channels.csv").open(newline="") as handle:
        table = {
            row["Channel"]: kind_of(row["Subsystem"], row["Physical Unit"])
            for row in csv.DictReader(handle)
        }
    kinds = [table[c.split("/")[1]] for c in channels]
    return Mission.dense(name, channels, kinds, hours, means)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mission", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--hourly", type=Path, required=True, help="telemetry_hourly_means's")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw" / "esa_ad")
    parser.add_argument("--days", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=int, required=True, help="the most hours per day")
    parser.add_argument("--library-end", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    mission = loaded(arguments.hourly, arguments.mission, arguments.raw)
    library = Library(
        loaded(arguments.hourly, arguments.library, arguments.raw), arguments.library_end
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    ends = block_ends(arguments.days)
    for days in arguments.days:
        start = days * HOURS_PER_DAY
        label = f"{arguments.mission}-N{days}"
        hours = drawn_hours(
            mission, start, ends[days] * HOURS_PER_DAY, arguments.windows, arguments.seed + days
        )
        ridges = {
            "ridge": fitted_ridges(mission, start, window=False),
            "window_ridge": fitted_ridges(mission, start, window=True),
        }
        cells = cells_of(mission, hours, ridges)
        write_cells(arguments.out, label, arguments.mission, cells)
        selections = selections_of(mission, cells, library, arguments.seed + 1000 + days)
        write_selections(arguments.out, label, arguments.mission, selections)
        print(
            f"{label}: {len(hours)} hours, {len(cells)} cells, ridges for "
            f"{len(ridges['ridge'])} channels, "
            f"{len(selections) // len(RUNGS)} cells with a counterpart"
        )
    (arguments.out / f"conditions-{arguments.mission}.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
