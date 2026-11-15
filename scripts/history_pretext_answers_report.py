r"""A forecasting pretext's answer for the hidden hour against answers with no model and a ridge.

A backbone trained to fill in each channel's last hour of a window joined with its history is
asked for that hour as the pretext hides it, and its decoder is scored beside four answers that
learnt nothing from it: the channel's last reading before the hour, carried forward; the median
of the channel's hourly means over the hours before; the hour one period back, the period being
the lag of up to half the window that best repeated the channel's own hours before, so that a
channel that cycles is answered by its last cycle; a ridge per channel over every feature of
the channel's 47 hours before, fitted on the months the pretraining trained on, which is the
classical forecaster given what the backbone is given for one channel; and, beside it, a ridge
per channel that also reads the hourly means of the window's other channels, since the backbone
attends across the window and a gain over the first ridge may be theirs. Each is scored on
every hidden cell that held a reading, by the objective's Huber loss over the cell's five values
in normalised units and by the squared error of its mean; a cell whose channel read nothing
earlier in the window is left out of all of them, and counted.

Three sets of months are read, each by the same code: the months the pretraining trained on
(``--side training``), the months its corpus held out to choose the epoch (``--side
validation``), and months of another publication the pretraining never read (``--side
validation --months KEY CHECKSUM``), accepted only when the run's corpus adopted that
publication's statistics, so the two lay the same reading into the same normalised value. Units
named in ``--frozen`` are left unread. A window whose history holds an hour in which no channel
read, as a month not loaded or a gap in the downlink leaves, is dropped before the draw.

Cells of one window share the window, and neighbouring windows share their hours, so the
intervals resample whole windows, and the decoder is compared with each answer on the same cells.
Each time axis of the corpus — a satellite's mission — is read on its own as well as together,
and the windows are drawn per axis, so a mission with fewer months is read on as many.

    uv run --env-file .env.r2 scripts/history_pretext_answers_report.py \\
        --result <key> <checksum> --corpus <key> <checksum> --side training \\
        --fit-ridge 20000 [--windows 1500] [--device mps] --out DIR
    uv run --env-file .env.r2 scripts/history_pretext_answers_report.py \\
        --result <key> <checksum> --corpus <key> <checksum> --side validation \\
        [--months <key> <checksum> --frozen UNITS.txt] --ridges DIR --out DIR

The first form fits both ridges on training windows outside the ones it then reads and writes
``ridge.npz`` and ``window-ridge.npz``; the second reads them from the directory the first wrote
to. Each writes ``answers-<label>.csv``, one row per scored cell,
and ``context-<label>.csv``, each cell's 47 hourly means for a forecaster outside this package,
and prints the tables.
"""

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the package is installed, and this file needs nothing of the other scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402

if TYPE_CHECKING:
    # The reading needs torch; scoring what it read does not, so the tests run without it.
    from emblema.pretraining.adapters.objective.pretext_reading import ReadCells
    from emblema.shared.adapters.arrays.joined_windows import JoinedWindows

ANSWERS = ("decoder", "last_reading", "median_of_history", "same_phase", "ridge", "window_ridge")
COLUMNS = (
    "window",
    "axis",
    "channel",
    "lag",
    "target_mean",
    *(f"huber_{answer}" for answer in ANSWERS),
    *(f"mean_{answer}" for answer in ANSWERS),
)
VALUES = BinnedTokens.VALUE_FEATURES
LAST = BinnedTokens.FEATURE_NAMES.index("last")
MEAN = BinnedTokens.FEATURE_NAMES.index("mean")
PRESENCE = BinnedTokens.PRESENCE
RESAMPLES = 2000
CHUNK = 16
READ_SEED = 1
FIT_SEED = 2
# A lag is a period only if it repeated the channel over this many earlier hours of the window.
MIN_PAIRS = 6


@dataclass(frozen=True)
class ScoredCell:
    """One hidden cell that held a reading, and each answer's loss on it.

    ``lag`` is the period the same-phase answer went back, or None where no lag repeated the
    channel over enough earlier hours and that answer fell back to the channel's latest hour.
    ``history`` is the channel's hourly mean over the hours before, NaN where it read nothing.
    """

    window: int
    axis: str
    channel: int
    target_mean: float
    losses: tuple[float, ...]
    means: tuple[float, ...]
    lag: int | None = None
    history: tuple[float, ...] = ()


@dataclass(frozen=True)
class Comparison:
    """The answers' mean losses on one set of cells, and the decoder's paired differences."""

    label: str
    windows: int
    cells: int
    means: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    differences: tuple[tuple[float, float, float], ...]


def huber(error: NDArray[np.float64], delta: float = 1.0) -> NDArray[np.float64]:
    magnitude = np.abs(error)
    return np.where(magnitude <= delta, 0.5 * error**2, delta * (magnitude - 0.5 * delta))


def same_phase(
    values: NDArray[np.float64], present: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], int | None]:
    """One channel's answer for its next hour: the hour one period back, and that period.

    ``values`` holds the channel's hours before the hidden one, oldest first, and ``present``
    which of them held a reading. Each lag up to half the window, the hidden hour counted, is
    scored by how well an hour that lag back answered every earlier hour both read, so that a
    48-hour window can find a daily cycle; the best lag whose hour back from the hidden one read
    gives the answer. Without such a lag the answer is the latest hour read.

    Raises:
        ValueError: If no hour read.
    """
    read = np.flatnonzero(present)
    if not len(read):
        raise ValueError("the channel read in no hour before the hidden one")
    hours = len(values)
    best: tuple[float, int] | None = None
    for lag in range(1, (hours + 1) // 2 + 1):
        if not present[hours - lag]:
            continue
        pairs = present[lag:] & present[:-lag]
        if pairs.sum() < MIN_PAIRS:
            continue
        error = float(huber(values[:-lag][pairs] - values[lag:][pairs]).mean())
        if best is None or error < best[0]:
            best = (error, lag)
    if best is None:
        return values[read[-1]], None
    return values[hours - best[1]], best[1]


@dataclass(frozen=True)
class FitRow:
    """One cell a ridge fits on or answers: its channel's history and the window's other channels.

    ``history`` is the channel's ``[hours, features]`` before the hidden hour; ``means`` the
    ``[hours, companions]`` hourly means of the window's other channels, ``companions`` their
    identifiers in the order the columns take, sorted; ``target`` the hidden hour's five values.
    """

    channel: int
    history: NDArray[np.float64]
    means: NDArray[np.float64]
    companions: NDArray[np.int64]
    target: NDArray[np.float64]
    half: int


@dataclass
class _Sums:
    """The sums a ridge is solved from, a fit's squared error on these rows, and the rows' range."""

    features: int
    rows: int = 0
    gram: NDArray[np.float64] = field(init=False)
    cross: NDArray[np.float64] = field(init=False)
    low: NDArray[np.float64] = field(init=False)
    high: NDArray[np.float64] = field(init=False)
    squares: float = 0.0

    def __post_init__(self) -> None:
        self.gram = np.zeros((self.features, self.features))
        self.cross = np.zeros((self.features, 5))
        self.low = np.full(self.features, np.inf)
        self.high = np.full(self.features, -np.inf)

    def add(self, described: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        self.rows += len(described)
        self.gram += described.T @ described
        self.cross += described.T @ targets
        self.low = np.minimum(self.low, described.min(axis=0))
        self.high = np.maximum(self.high, described.max(axis=0))
        self.squares += float((targets * targets).sum())

    def plus(self, other: "_Sums") -> "_Sums":
        total = _Sums(self.features, self.rows + other.rows)
        total.gram = self.gram + other.gram
        total.cross = self.cross + other.cross
        total.low = np.minimum(self.low, other.low)
        total.high = np.maximum(self.high, other.high)
        total.squares = self.squares + other.squares
        return total

    def solved(self, penalty: float) -> NDArray[np.float64]:
        """The ridge on these rows, its intercept (the first feature) unpenalised."""
        diagonal = np.full(self.features, penalty)
        # The intercept takes no penalty; a whisker keeps a feature never seen solvable.
        diagonal[0] = 1e-9
        solved: NDArray[np.float64] = np.linalg.solve(self.gram + np.diag(diagonal), self.cross)
        return solved

    def missed(self, weights: NDArray[np.float64]) -> float:
        """The squared error of ``weights`` over these rows, from their sums alone."""
        return float(
            self.squares
            - 2.0 * np.sum(weights * self.cross)
            + np.sum(weights * (self.gram @ weights))
        )


@dataclass(frozen=True)
class ChannelRidge:
    """A ridge per channel forecasting a cell's five values from the history it is given.

    The history is the channel's cells over the hours before the hidden one, every feature of
    each — presence, count, the five values and the two offsets — flattened in time order, with
    an intercept; a bin that held no reading is the zeros the layout gives it. A ridge over the
    window reads beside these the hourly means of the window's other channels, so that a gain
    over it is not what another channel's hours tell. Each channel has its own ridge, fitted on
    the cells scored on the fitting windows, its penalty the one whose ridges fitted on either
    half of the fitting months miss the other half's windows least in squared error, the ridge
    kept then fitted on both halves, as the package's linear answer chooses; the halves are
    months, not windows, since neighbouring windows share their hours. An answer clips each
    feature to the range the fit saw, so a ridge never extrapolates: a linear rule is a fit
    inside the hull of its rows, and outside it one unusual hour threw an answer tens of units
    off in the trial. The penalty is chosen on the other half's rows unclipped, since the sums
    a half is kept as cannot be clipped after the fact; the clipping acts on the answer only. A
    window that lacks a channel the ridge over the window was fitted with gives it the zeros
    the layout gives an absent hour; a channel never fitted with is refused.

    Attributes:
        hours: Hours of history each ridge reads.
        corpus: Hex digest of the block the fitting windows were read from.
        window: Whether the other channels' hourly means join the features.
        weights: ``{channel: [features, 5]}``.
        penalties: ``{channel: penalty}`` chosen.
        rows: ``{channel: cells fitted}``.
        companions: ``{channel: [companions]}`` the channels whose means its ridge reads.
        low: ``{channel: [features]}`` the least value of each feature over the fit.
        high: ``{channel: [features]}`` the greatest.
    """

    PENALTIES: ClassVar[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    FLUSH_ENTRIES: ClassVar[int] = 1 << 24

    hours: int
    corpus: str
    window: bool = False
    weights: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    penalties: dict[int, float] = field(default_factory=dict)
    rows: dict[int, int] = field(default_factory=dict)
    companions: dict[int, NDArray[np.int64]] = field(default_factory=dict)
    low: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    high: dict[int, NDArray[np.float64]] = field(default_factory=dict)

    def features_of(self, companions: int) -> int:
        return 1 + self.hours * (BinnedTokens.N_FEATURES + (companions if self.window else 0))

    def described(self, row: FitRow) -> NDArray[np.float64]:
        """``[features]`` — the history flat after a one, the other channels' means after it.

        Raises:
            ValueError: If the history is not over the hours the ridge reads, or the means are
                not over them and the row's companions.
        """
        if row.history.shape != (self.hours, BinnedTokens.N_FEATURES):
            raise ValueError(
                f"the ridge reads {self.hours} hours of {BinnedTokens.N_FEATURES} features, "
                f"not {row.history.shape}"
            )
        if not self.window:
            return np.concatenate(([1.0], row.history.reshape(-1)))
        if row.means.shape != (self.hours, len(row.companions)):
            raise ValueError(
                f"the ridge reads {self.hours} hours of {len(row.companions)} other channels, "
                f"not {row.means.shape}"
            )
        return np.concatenate(([1.0], row.history.reshape(-1), row.means.reshape(-1)))

    def aligned(self, row: FitRow) -> FitRow:
        """``row`` with its means laid over the companions its channel's ridge knows.

        A companion the window lacks takes a column of zeros, as the layout gives an absent
        hour; the row is returned as it is where the ridge reads the channel alone or the
        companions already agree.

        Raises:
            ValueError: If the window holds a channel the ridge was not fitted with.
        """
        known = self.companions.get(row.channel)
        if not self.window or known is None or np.array_equal(row.companions, known):
            return row
        places = {int(c): k for k, c in enumerate(known)}
        strange = [int(c) for c in row.companions if int(c) not in places]
        if strange:
            raise ValueError(
                f"channel {row.channel}'s ridge was not fitted with channels {strange}, which "
                "the window holds"
            )
        means = np.zeros((row.means.shape[0], len(known)))
        for column, companion in enumerate(row.companions):
            means[:, places[int(companion)]] = row.means[:, column]
        return FitRow(row.channel, row.history, means, known, row.target, row.half)

    def answer(self, row: FitRow) -> NDArray[np.float64]:
        """The channel's five values for the hidden hour, each feature clipped to the fit's range.

        Raises:
            ValueError: If no ridge was fitted for the channel, or the window holds a channel
                its ridge was not fitted with.
        """
        channel = row.channel
        if channel not in self.weights:
            raise ValueError(f"no ridge was fitted for channel {channel}")
        row = self.aligned(row)
        described = np.clip(self.described(row), self.low[channel], self.high[channel])
        answered: NDArray[np.float64] = described @ self.weights[channel]
        return answered

    @classmethod
    def fitted(
        cls, hours: int, corpus: str, rows: Iterable[FitRow], *, window: bool = False
    ) -> "ChannelRidge":
        """The ridges fitted on ``rows``; see ``RidgeFit`` for the choice of penalty."""
        fit = RidgeFit(cls(hours, corpus, window))
        for row in rows:
            fit.add(row)
        return fit.fitted()

    def save(self, path: Path) -> None:
        arrays: dict[str, NDArray[np.float64] | NDArray[np.int64] | NDArray[np.str_]] = {
            "hours": np.array([self.hours]),
            "corpus": np.array([self.corpus]),
            "window": np.array([int(self.window)]),
            "channels": np.array(sorted(self.weights), dtype=np.int64),
        }
        for channel, weights in self.weights.items():
            arrays[f"weights_{channel}"] = weights
            arrays[f"penalty_{channel}"] = np.array([self.penalties[channel]])
            arrays[f"rows_{channel}"] = np.array([self.rows[channel]], dtype=np.int64)
            arrays[f"companions_{channel}"] = self.companions[channel]
            arrays[f"low_{channel}"] = self.low[channel]
            arrays[f"high_{channel}"] = self.high[channel]
        np.savez(path, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as arrays:
            ridge = cls(
                int(arrays["hours"][0]), str(arrays["corpus"][0]), bool(arrays["window"][0])
            )
            for channel in arrays["channels"].tolist():
                ridge.weights[channel] = arrays[f"weights_{channel}"]
                ridge.penalties[channel] = float(arrays[f"penalty_{channel}"][0])
                ridge.rows[channel] = int(arrays[f"rows_{channel}"][0])
                ridge.companions[channel] = arrays[f"companions_{channel}"]
                ridge.low[channel] = arrays[f"low_{channel}"]
                ridge.high[channel] = arrays[f"high_{channel}"]
        return ridge

    def table(self) -> str:
        counts = {p: sum(1 for c in self.penalties.values() if c == p) for p in self.PENALTIES}
        chosen = ", ".join(f"{p:g}: {n}" for p, n in counts.items() if n)
        edges = sorted(
            c for c, p in self.penalties.items() if p in (self.PENALTIES[0], self.PENALTIES[-1])
        )
        features = sorted({w.shape[0] for w in self.weights.values()})
        reads = "the window's other channels beside" if self.window else "the channel alone"
        return (
            f"ridge over {self.hours} hours, {reads}, {' or '.join(map(str, features))} "
            f"features, {len(self.weights)} channels, {sum(self.rows.values())} cells fitted; "
            f"penalties chosen: {chosen}; channels at the grid's edges: "
            f"{', '.join(map(str, edges)) or 'none'}"
        )


@dataclass
class RidgeFit:
    """A ridge's sums taken row by row, so two ridges can fit on one pass through the windows.

    A channel seen in one half only takes the penalty in the middle of the grid, since no other
    half can choose it. Rows wait in memory until their entries reach ``FLUSH_ENTRIES``, then
    their sums are taken. The first row of a channel fixes the companions its ridge over the
    window reads; a later window lacking one of them gives it zeros, as the layout gives an
    absent hour.
    """

    ridge: ChannelRidge
    sums: list[dict[int, _Sums]] = field(default_factory=lambda: [{}, {}])
    pending: list[dict[int, list[FitRow]]] = field(default_factory=lambda: [{}, {}])
    entries: list[int] = field(default_factory=lambda: [0, 0])

    def add(self, row: FitRow) -> None:
        """Takes one row, laid over the companions the channel's first row had.

        Raises:
            ValueError: If the row's window holds a channel the channel's rows before did not.
        """
        if row.channel not in self.ridge.companions:
            self.ridge.companions[row.channel] = row.companions
        row = self.ridge.aligned(row)
        self.pending[row.half].setdefault(row.channel, []).append(row)
        self.entries[row.half] += self.ridge.features_of(len(row.companions))
        if self.entries[row.half] >= self.ridge.FLUSH_ENTRIES:
            self._flush(row.half)

    def _flush(self, half: int) -> None:
        for channel, rows in self.pending[half].items():
            described = np.stack([self.ridge.described(row) for row in rows])
            targets = np.stack([row.target for row in rows])
            if channel not in self.sums[half]:
                self.sums[half][channel] = _Sums(self.ridge.features_of(len(rows[0].companions)))
            self.sums[half][channel].add(described, targets)
        self.pending[half].clear()
        self.entries[half] = 0

    def fitted(self) -> ChannelRidge:
        self._flush(0)
        self._flush(1)
        ridge = self.ridge
        grid = ridge.PENALTIES
        for channel in sorted(set(self.sums[0]) | set(self.sums[1])):
            first, second = self.sums[0].get(channel), self.sums[1].get(channel)
            if first is not None and second is not None:
                both = first.plus(second)
                fit_a, fit_b = first, second
                penalty = min(
                    grid,
                    key=lambda p: fit_b.missed(fit_a.solved(p)) + fit_a.missed(fit_b.solved(p)),
                )
            else:
                both = first if first is not None else self.sums[1][channel]
                penalty = grid[len(grid) // 2]
            ridge.weights[channel] = both.solved(penalty)
            ridge.penalties[channel] = penalty
            ridge.rows[channel] = both.rows
            ridge.low[channel] = both.low
            ridge.high[channel] = both.high
        return ridge


def laid_cells(read: "ReadCells", *, bins: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``[bins, rows, features]`` of every reading and the channel of each row.

    Raises:
        ValueError: If the cells are not laid bin by bin over the same channels.
    """
    timed = ~read.timeless
    channel_ids = read.channel_ids[timed]
    if len(channel_ids) % bins:
        raise ValueError(f"{len(channel_ids)} cells do not lie over {bins} bins")
    rows = len(channel_ids) // bins
    by_bin = channel_ids.reshape(bins, rows)
    if (by_bin != by_bin[0]).any():
        raise ValueError("the cells are not laid bin by bin over the same channels")
    return read.full[timed].reshape(bins, rows, -1), by_bin[0]


def fit_row(
    full: NDArray[np.float64], channels: NDArray[np.int64], row: int, half: int = 0
) -> FitRow:
    """The cell of ``row`` with its history and the other channels' hourly means, by channel."""
    others = np.array([k for k in np.argsort(channels, kind="stable") if k != row])
    return FitRow(
        channel=int(channels[row]),
        history=full[:-1, row, :],
        means=full[:-1, others, MEAN],
        companions=channels[others].astype(np.int64),
        target=full[-1, row, VALUES],
        half=half,
    )


def fitting_rows(read: "ReadCells", *, bins: int, half: int) -> list[FitRow]:
    """The window's cells a ridge fits on: those the scoring would score, with their history."""
    full, channels = laid_cells(read, bins=bins)
    earlier = full[:-1, :, PRESENCE] > 0
    return [
        fit_row(full, channels, int(row), half)
        for row in np.flatnonzero(full[-1, :, PRESENCE] > 0)
        if earlier[:, row].any()
    ]


def scored(
    read: "ReadCells",
    *,
    bins: int,
    window: int,
    axis: str,
    ridge: ChannelRidge,
    window_ridge: ChannelRidge,
    delta: float = 1.0,
) -> tuple[list[ScoredCell], int]:
    """The window's hidden cells scored by each answer, and how many had no earlier reading.

    Raises:
        ValueError: If the cells are not laid bin by bin over the same channels, the draw hid
            other than the whole last bin, or the decoder gave no answers.
    """
    if read.answers is None:
        raise ValueError("the draw was asked for without the decoder's answers")
    full, channels = laid_cells(read, bins=bins)
    timed = ~read.timeless
    hidden = read.hidden[timed].reshape(bins, len(channels))
    if not hidden[-1].all() or hidden[:-1].any():
        raise ValueError("the draw hid other than the whole last bin")
    answers = read.answers[timed].reshape(bins, len(channels), -1)
    earlier = full[:-1, :, PRESENCE] > 0
    cells: list[ScoredCell] = []
    unscored = 0
    for row in np.flatnonzero(full[-1, :, PRESENCE] > 0):
        seen = np.flatnonzero(earlier[:, row])
        if not len(seen):
            unscored += 1
            continue
        target = full[-1, row, VALUES]
        cycled, lag = same_phase(full[:-1, row, VALUES], earlier[:, row])
        asked = fit_row(full, channels, int(row))
        answered = (
            answers[-1, row],
            np.full(5, full[seen[-1], row, LAST]),
            np.full(5, np.median(full[seen, row, MEAN])),
            cycled,
            ridge.answer(asked),
            window_ridge.answer(asked),
        )
        history = np.where(earlier[:, row], full[:-1, row, MEAN], np.nan)
        cells.append(
            ScoredCell(
                window=window,
                axis=axis,
                channel=asked.channel,
                target_mean=float(target[0]),
                losses=tuple(float(huber(a - target, delta).mean()) for a in answered),
                means=tuple(float(a[0]) for a in answered),
                lag=lag,
                history=tuple(float(h) for h in history),
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
        means=tuple(float(m) for m in point),
        intervals=tuple((float(a), float(b)) for a, b in zip(low, high, strict=True)),
        differences=tuple(differences),
    )


def table(comparisons: Sequence[Comparison]) -> str:
    heads = ["Decoder", "Last reading", "Median of history", "Same phase", "Ridge", "Window ridge"]
    lines = [
        "| Cells | Windows | Cells scored | "
        + " | ".join(heads)
        + " | "
        + " | ".join(f"Decoder - {h.lower()}" for h in heads[1:])
        + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in range(2 * len(heads) - 1)) + " |",
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


def lags(cells: Sequence[ScoredCell]) -> str:
    """Per axis, how often the same-phase answer fell back, and the periods it chose most."""
    lines = []
    for axis in sorted({cell.axis for cell in cells}):
        chosen = [cell.lag for cell in cells if cell.axis == axis]
        fitted = np.array([lag for lag in chosen if lag is not None])
        ids, counts = np.unique(fitted, return_counts=True)
        top = ", ".join(
            f"{ids[i]} h {counts[i] / len(chosen):.0%}" for i in np.argsort(-counts)[:5]
        )
        lines.append(
            f"{axis}: {len(chosen) - len(fitted)} of {len(chosen)} cells fell back; "
            f"periods chosen most: {top or 'none'}"
        )
    return "\n".join(lines)


def write_cells(path: Path, cells: Sequence[ScoredCell]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for cell in cells:
            lag = "" if cell.lag is None else cell.lag
            writer.writerow(
                (
                    cell.window,
                    cell.axis,
                    cell.channel,
                    lag,
                    cell.target_mean,
                    *cell.losses,
                    *cell.means,
                )
            )


def write_context(path: Path, cells: Sequence[ScoredCell], hours: int) -> None:
    """Each scored cell's hourly means before the hidden hour, blank where it read nothing."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("window", "axis", "channel", *(f"h{h:02d}" for h in range(1, hours + 1))))
        for cell in cells:
            if len(cell.history) != hours:
                raise ValueError(f"a cell holds {len(cell.history)} hours of history, not {hours}")
            hours_read = ("" if np.isnan(h) else f"{h:.6g}" for h in cell.history)
            writer.writerow((cell.window, cell.axis, cell.channel, *hours_read))


@dataclass(frozen=True)
class WindowSet:
    """The windows of one side of a publication, joined with their history, complete ones only.

    Attributes:
        joined: Every loaded window joined with the windows before it.
        targets: Positions in ``joined`` of the side's windows whose history is complete.
        axes: The axis of each target.
        dropped: The side's windows dropped for a history with an hour nobody read.
        units: The unit (a mission's month) of each target, where the caller gives them.
    """

    joined: "JoinedWindows"
    targets: NDArray[np.int64]
    axes: tuple[str, ...]
    dropped: int
    units: tuple[str, ...] = ()

    def draw(self, per_axis: int, seed: int, excluding: Sequence[int] = ()) -> NDArray[np.int64]:
        """Up to ``per_axis`` targets of each axis, drawn without replacement, sorted."""
        rng = np.random.default_rng(seed)
        axes = np.array(self.axes)
        out = {int(p) for p in excluding}
        chosen: list[int] = []
        for axis in sorted(set(self.axes)):
            pool = np.array([p for p in self.targets[axes == axis] if int(p) not in out])
            size = min(per_axis, len(pool))
            chosen.extend(int(p) for p in rng.choice(pool, size=size, replace=False))
        return np.array(sorted(chosen), dtype=np.int64)

    def complete(self, position: int) -> bool:
        cells = self.joined[int(position)].cells
        hourly: NDArray[np.bool_] = cells[:, :-1, :].any(axis=(0, 2))
        return bool(hourly.all())


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the store
    from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.pretraining.adapters.documents.pretraining_result_json import (
        PretrainingResultJson,
    )
    from emblema.pretraining.adapters.objective.pretext_reading import PretextReading
    from emblema.pretraining.adapters.training.trained_model import TrainedModel
    from emblema.pretraining.domain.training.objective_loss import LossKind
    from emblema.shared.adapters.arrays.joined_windows import JoinedWindows
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum
    from emblema.shared.kernel.ordering import seeded_rank

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--corpus", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--side", choices=("validation", "training"), default="validation")
    parser.add_argument("--months", nargs=2, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--frozen", type=Path, help="units left unread, one per line")
    parser.add_argument("--windows", type=int, default=1500, help="windows drawn per axis")
    parser.add_argument("--fit-ridge", type=int, metavar="N", help="training windows per axis")
    parser.add_argument("--ridges", type=Path, help="the directory --fit-ridge wrote the ridges to")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--label", help="suffix of the files written; the side by default")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if (arguments.fit_ridge is None) == (arguments.ridges is None):
        raise SystemExit("give --fit-ridge N to fit the ridges or --ridges DIR to read them")
    if arguments.months and arguments.side != "validation":
        raise SystemExit("another publication is read on its validation side only")
    if arguments.months and arguments.fit_ridge is not None:
        raise SystemExit("the ridge is fitted on the run's own publication, not on another's")

    store = configured_store(Settings())
    blocks = PublishedCorpusBlocks(store, arguments.workspace)
    result = PretrainingResultJson().decode(
        store.get(ArtifactRef(arguments.result[0], Checksum.parse(arguments.result[1])))
    )
    trained = TrainedModel.read(store.get(result.outcome.backbone))
    bins, history = trained.architecture.bins, trained.architecture.history
    if trained.loss is None or trained.loss.kind is not LossKind.HUBER:
        raise SystemExit("the run did not score its misses by a Huber loss")
    delta = trained.loss.huber_delta
    (read_shape,) = result.mixture.corpora
    own_ref = ArtifactRef(arguments.corpus[0], Checksum.parse(arguments.corpus[1]))
    own = blocks.manifest_of(own_ref)
    if own.block.checksum != read_shape.checksum:
        raise SystemExit(f"{arguments.corpus[0]} is not the corpus the run read")

    if arguments.months:
        months_ref = ArtifactRef(arguments.months[0], Checksum.parse(arguments.months[1]))
        if own.statistics_from != months_ref:
            raise SystemExit(
                "the run's corpus did not adopt the statistics of the publication asked for, "
                "so its values would not lie in the backbone's units"
            )
        published = blocks.manifest_of(months_ref)
    else:
        published = own
    frozen: set[str] = set()
    if arguments.frozen:
        frozen = {
            line.strip() for line in arguments.frozen.read_text().splitlines() if line.strip()
        }
        unknown = frozen - set(published.units) - set(published.empty_units)
        if unknown:
            raise SystemExit(f"units to leave unread that the publication does not name: {unknown}")
    block = blocks.block_of(published)
    positions = {key: index for index, key in enumerate(published.units)}
    loaded = sorted(
        positions[key]
        for key in (*published.training_units, *published.validation_units)
        if key in positions and key not in frozen
    )
    side_units = (
        published.training_units if arguments.side == "training" else published.validation_units
    )
    side_positions = {
        positions[key] for key in side_units if key in positions and key not in frozen
    }
    at = {unit: index for index, unit in enumerate(loaded)}
    extents = [block.extent_of(unit_index) for unit_index in range(len(block))]
    window_positions = [i for i in range(len(block)) if block.unit_of(i) in at]
    axes_of = [
        PublishedCorpusManifest.axis_of(published.units[block.unit_of(i)]) for i in window_positions
    ]
    joined = JoinedWindows.of(
        (block.at(window_positions),),
        axes=axes_of,
        starts=[extents[i][0] for i in window_positions],
        ends=[extents[i][1] for i in window_positions],
        history=history,
        bins=bins,
    )

    def window_set(unit_positions: set[int]) -> WindowSet:
        candidates = [
            k for k, i in enumerate(window_positions) if block.unit_of(i) in unit_positions
        ]
        partial = WindowSet(
            joined, np.array(candidates, dtype=np.int64), tuple(axes_of[k] for k in candidates), 0
        )
        kept = [k for k in candidates if partial.complete(k)]
        return WindowSet(
            joined,
            np.array(kept, dtype=np.int64),
            tuple(axes_of[k] for k in kept),
            len(candidates) - len(kept),
            tuple(published.units[block.unit_of(window_positions[k])] for k in kept),
        )

    reading_set = window_set(side_positions)
    chosen = reading_set.draw(arguments.windows, READ_SEED)
    label = arguments.label or arguments.side
    print(
        f"{len(chosen)} {arguments.side} windows drawn, up to {arguments.windows} per axis; "
        f"{reading_set.dropped} dropped for an hour nobody read, {len(frozen)} units left unread"
    )
    arguments.out.mkdir(parents=True, exist_ok=True)

    if arguments.fit_ridge is not None:
        training_positions = {positions[key] for key in own.training_units if key in positions}
        fitting_set = (
            reading_set
            if arguments.side == "training" and not arguments.months
            else window_set(training_positions)
        )
        # The training-side reading of the same size stays outside the fit, whichever side this is.
        read_on_training = fitting_set.draw(arguments.windows, READ_SEED)
        fit_windows = fitting_set.draw(
            arguments.fit_ridge, FIT_SEED, excluding=read_on_training.tolist()
        )
        print(
            f"fitting the ridge on {len(fit_windows)} training windows outside the "
            f"{len(read_on_training)} read"
        )
        axis_at = dict(zip(fitting_set.targets.tolist(), fitting_set.axes, strict=True))
        unit_at = dict(zip(fitting_set.targets.tolist(), fitting_set.units, strict=True))
        read_set = set(read_on_training.tolist())
        for axis in sorted(set(fitting_set.axes)):
            pool = sum(1 for p, a in axis_at.items() if a == axis and p not in read_set)
            fitted = sum(1 for p in fit_windows.tolist() if axis_at[p] == axis)
            print(f"{axis}: {pool} training windows outside the ones read, {fitted} fitted")
        # Halves by month, as the package's linear answer draws them: neighbouring windows share
        # their hours, so halves by window would score a fit on nearly itself.
        half_of = {
            unit: seeded_rank(FIT_SEED, "half", unit)[0] & 1 for unit in set(fitting_set.units)
        }
        halves = [half_of[unit_at[p]] for p in fit_windows.tolist()]
        print(
            f"months in either half: {sum(1 for h in half_of.values() if h == 0)} and "
            f"{sum(1 for h in half_of.values() if h == 1)}"
        )
        reading = PretextReading.of(trained, device=arguments.device)

        def rows() -> Iterable[FitRow]:
            for start in range(0, len(fit_windows), CHUNK):
                batch = fit_windows[start : start + CHUNK].tolist()
                asked = joined.at(batch)
                drawn = reading.drawn(
                    [asked[i] for i in range(len(asked))],
                    [0] * len(asked),
                    answered=False,
                    tail=True,
                )
                for k, read in enumerate(drawn):
                    yield from fitting_rows(read, bins=bins, half=halves[start + k])

        # One pass through the model feeds both ridges: reading the windows is the cost.
        digest = own.block.checksum.digest
        fits = (
            RidgeFit(ChannelRidge(bins - 1, digest)),
            RidgeFit(ChannelRidge(bins - 1, digest, window=True)),
        )
        for row in rows():
            for fit in fits:
                fit.add(row)
        ridge, window_ridge = (fit.fitted() for fit in fits)
        ridge.save(arguments.out / "ridge.npz")
        window_ridge.save(arguments.out / "window-ridge.npz")
    else:
        ridge = ChannelRidge.load(arguments.ridges / "ridge.npz")
        window_ridge = ChannelRidge.load(arguments.ridges / "window-ridge.npz")
        if ridge.window or not window_ridge.window:
            raise SystemExit(
                "ridge.npz must be the ridge over the channel, window-ridge.npz the other"
            )
        for one in (ridge, window_ridge):
            if one.corpus != own.block.checksum.digest:
                raise SystemExit("a ridge was fitted on another corpus than the run read")
        reading = PretextReading.of(trained, device=arguments.device)
    print(ridge.table())
    print(window_ridge.table())

    for one in (ridge, window_ridge):
        unfitted = sorted(set(joined[0].rows.tolist()) - set(one.weights))
        if unfitted:
            raise SystemExit(f"the windows hold channels a ridge was not fitted for: {unfitted}")
    cells: list[ScoredCell] = []
    unscored = 0
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start : start + CHUNK].tolist()
        asked = joined.at(batch)
        drawn = reading.drawn(
            [asked[i] for i in range(len(asked))], [0] * len(asked), answered=True, tail=True
        )
        for position, read in zip(batch, drawn, strict=True):
            axis = reading_set.axes[int(np.searchsorted(reading_set.targets, position))]
            found, missed = scored(
                read,
                bins=bins,
                window=position,
                axis=axis,
                ridge=ridge,
                window_ridge=window_ridge,
                delta=delta,
            )
            cells.extend(found)
            unscored += missed
    write_cells(arguments.out / f"answers-{label}.csv", cells)
    write_context(arguments.out / f"context-{label}.csv", cells, bins - 1)
    rng = np.random.default_rng(READ_SEED)
    print(
        f"{len(chosen)} windows, {len(cells)} hidden cells scored, {unscored} with no earlier "
        f"reading in the window; Huber (delta {delta:g}) over the five values, 95 % intervals "
        f"from {RESAMPLES} resamples of whole windows"
    )
    print(table(comparisons_of(cells, rng)))
    print(lags(cells))


if __name__ == "__main__":  # pragma: no cover
    main()
