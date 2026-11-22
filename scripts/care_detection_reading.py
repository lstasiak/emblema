r"""Whether a detector's scores set the benchmark's faults apart from its quiet excerpts.

An excerpt of the fault benchmark holds one event, the lead-up to a fault or a stretch without
one, so the reading ranks excerpts. An excerpt's statistic is how unusual its event looks beside
its own year: the largest median of the hourly score over twelve hours running inside the event,
read as the share of the blocks of the event's length, slid over the year the detector was
fitted on, that hold a median as large. Twelve hours is the span the benchmark's own counter
needs to call a fault found, and the median over them asks that most of those hours stand high,
as that counter does, so one hour's spike moves nothing. Reading the event against blocks of its
own length takes the length out, which nearly spells the label here, and reading it against its
own year takes the turbine's level out. An event no block reaches ties with every other such.

The verdict is the area under the curve of that statistic between the excerpts with a fault and
those without, pairs taken inside a farm only, for the best detector of a family; its chance is
the same best taken with the labels shuffled inside each farm. The turbines are too few for an
interval that covers as it claims, so the shuffle alone decides and the intervals beside it are
approximate.

    uv run scripts/care_detection_reading.py power --root data/raw/care/CARE_To_Compare \\
        [--draws 20000] [--seed 1]

``power`` reads only the benchmark's tables of events: on the layout of the excerpts read it
draws scores with no fault in them and with a planted one, and prints what area a family's best
must reach and how often a true area reaches it.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.care_excerpts import Event, read_events  # noqa: E402
from scripts.reporting import table  # noqa: E402

Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]
Ints = NDArray[np.int64]

# The hours the benchmark's counter needs to call a fault found: 72 rows of ten minutes.
SPAN = 12
# The fewest scored hours of a span whose median is read.
LEAST_SCORED = 9
SHUFFLES = 10_000
# One-sided, over the two families read, the smaller p-value first.
LEVEL = 0.025
FAMILIES = 2


def running_medians(scores: Floats) -> Floats:
    """Per hour, the median of the scores of the span ending at it.

    Missing where the span is not whole or holds too few scored hours.
    """
    medians = np.full(len(scores), np.nan)
    if len(scores) < SPAN:
        return medians
    spans = sliding_window_view(scores, SPAN)
    scored = (~np.isnan(spans)).sum(axis=1) >= LEAST_SCORED
    with np.errstate(invalid="ignore"):
        values = np.nanmedian(np.where(scored[:, None], spans, 0.0), axis=1)
    medians[SPAN - 1 :] = np.where(scored, values, np.nan)
    return medians


def largest_in_blocks(medians: Floats, length: int) -> Floats:
    """The largest median of every block of ``length`` hours, the blocks with none left out.

    A block's spans are those lying wholly inside it, so a block of ``length`` hours reads
    ``length - SPAN + 1`` medians.
    """
    reach = length - SPAN + 1
    inside = medians[SPAN - 1 :]
    if reach < 1 or len(inside) < reach:
        return np.zeros(0)
    blocks = sliding_window_view(np.where(np.isnan(inside), -np.inf, inside), reach)
    largest: Floats = blocks.max(axis=1)
    return largest[np.isfinite(largest)]


def unusualness(scores: Floats, *, fitted: int, first: int, last: int) -> float | None:
    """How unusual an event's scores are beside the year before it.

    One less the share of the year's blocks of the event's length that reach the event's
    largest running median.

    Args:
        scores: The detector's hourly scores over the excerpt, missing where an hour is not
            scored.
        fitted: How many hours, from the first, are the year the detector was fitted on.
        first: The event's first hour.
        last: The event's last hour.

    Returns:
        ``None`` where the event holds no span scored, or the year no block of its length.

    Raises:
        ValueError: If the event starts inside the fitted year or lies outside the scores.
    """
    if not fitted <= first <= last < len(scores):
        raise ValueError(
            f"the event's hours {first} to {last} do not follow {fitted} fitted hours of "
            f"{len(scores)}"
        )
    length = last - first + 1
    event = largest_in_blocks(running_medians(scores[first : last + 1]), length)
    year = largest_in_blocks(running_medians(scores[:fitted]), length)
    if not len(event) or not len(year):
        return None
    return 1.0 - float((year >= event[0]).mean())


def ranks_within(values: Floats) -> Floats:
    """Each value's rank among its row, equal values sharing the mean of their ranks."""
    order = np.argsort(values, axis=-1, kind="stable")
    ranked = np.take_along_axis(values, order, axis=-1)
    ranks = np.empty_like(values, dtype=np.float64)
    positions = np.arange(1, values.shape[-1] + 1, dtype=np.float64)
    for row in np.ndindex(values.shape[:-1]):
        line = ranked[row]
        starts = np.flatnonzero(np.r_[True, line[1:] != line[:-1]])
        ends = np.r_[starts[1:], len(line)]
        shared = np.repeat((positions[starts] + positions[ends - 1]) / 2.0, ends - starts)
        ranks[row][order[row]] = shared
    return ranks


@dataclass(frozen=True)
class Ranking:
    """Excerpts ranked by several detectors, their farms and labels known.

    Attributes:
        statistics: ``[detectors, excerpts]``, larger where the event looks less like the year.
        anomaly: Per excerpt, whether it holds a fault.
        farms: Per excerpt, its farm; pairs are counted and labels shuffled inside a farm.
    """

    statistics: Floats
    anomaly: Bools
    farms: Sequence[str]

    def _strata(self) -> list[Ints]:
        names = np.asarray(self.farms)
        return [np.flatnonzero(names == farm) for farm in dict.fromkeys(self.farms)]

    def _pairs(self) -> int:
        return sum(
            int(self.anomaly[stratum].sum()) * int((~self.anomaly[stratum]).sum())
            for stratum in self._strata()
        )

    def areas(self) -> Floats:
        """Per detector, the share of pairs of one farm, a fault and none, ranked rightly."""
        won = np.zeros(len(self.statistics))
        for stratum in self._strata():
            faults = self.anomaly[stratum]
            ranks = ranks_within(self.statistics[:, stratum])
            count = int(faults.sum())
            won += ranks[:, faults].sum(axis=1) - count * (count + 1) / 2.0
        areas: Floats = won / self._pairs()
        return areas

    def shuffled_areas(self, shuffles: int, seed: int) -> Floats:
        """``[shuffles, detectors]``: the areas with the labels shuffled inside each farm."""
        draws = np.random.default_rng(seed)
        won = np.zeros((shuffles, len(self.statistics)))
        for stratum in self._strata():
            faults = self.anomaly[stratum]
            ranks = ranks_within(self.statistics[:, stratum])
            count = int(faults.sum())
            drawn = draws.permuted(np.tile(faults, (shuffles, 1)), axis=1)
            won += drawn.astype(np.float64) @ ranks.T - count * (count + 1) / 2.0
        shuffled: Floats = won / self._pairs()
        return shuffled

    def best_beyond_chance(self, shuffles: int = SHUFFLES, seed: int = 1) -> tuple[float, float]:
        """The best detector's area, and how often a shuffle's best reaches it."""
        best = float(self.areas().max())
        reached = (self.shuffled_areas(shuffles, seed).max(axis=1) >= best - 1e-12).sum()
        return best, float((1 + reached) / (1 + shuffles))


def confirmed(p_values: Sequence[float], level: float = LEVEL) -> list[bool]:
    """Which of a family of tests pass, the smallest p-value held to the strictest level."""
    order = sorted(range(len(p_values)), key=lambda test: p_values[test])
    passed = [False] * len(p_values)
    for step, test in enumerate(order):
        if p_values[test] > level / (len(p_values) - step):
            break
        passed[test] = True
    return passed


# --- power -----------------------------------------------------------------------------------


def shift_of(area: float) -> float:
    """The shift between two unit normal scores whose area under the curve is ``area``."""
    return math.sqrt(2.0) * NormalDist().inv_cdf(area)


@dataclass(frozen=True)
class Layout:
    """The excerpts a reading ranks: their farms, turbines and labels, and no score."""

    farms: tuple[str, ...]
    turbines: tuple[int, ...]
    anomaly: Bools

    @classmethod
    def of(cls, events: Sequence[Event]) -> Layout:
        ranked = [event for event in events if event.ranked]
        return cls(
            farms=tuple(event.farm for event in ranked),
            turbines=tuple(event.turbine for event in ranked),
            anomaly=np.asarray([event.anomaly for event in ranked], dtype=np.bool_),
        )

    def drawn(
        self,
        draws: np.random.Generator,
        *,
        detectors: int,
        shifts: Floats,
        shared: float,
        turbine: float,
    ) -> Ranking:
        """Scores drawn on the layout.

        Args:
            draws: The source of the draws.
            detectors: How many detectors rank the excerpts.
            shifts: Per detector, how far a fault's score lies above a quiet excerpt's.
            shared: The share of a score's variance all detectors share on an excerpt.
            turbine: The share of it an excerpt shares with its turbine's other excerpts.
        """
        excerpts = len(self.anomaly)
        keys = [f"{farm}-{name}" for farm, name in zip(self.farms, self.turbines, strict=True)]
        index = np.unique(keys, return_inverse=True)[1]
        of_turbine = draws.standard_normal(index.max() + 1)[index]
        common = draws.standard_normal(excerpts)
        own = draws.standard_normal((detectors, excerpts))
        rest = 1.0 - shared - turbine
        scores = (
            math.sqrt(turbine) * of_turbine
            + math.sqrt(shared) * common
            + math.sqrt(rest) * own
            + shifts[:, None] * self.anomaly
        )
        return Ranking(statistics=scores, anomaly=self.anomaly, farms=self.farms)


def best_areas(
    layout: Layout, source: np.random.Generator, *, draws: int, shifts: Floats, shared: float
) -> Floats:
    """The best detector's area in each of ``draws`` sets of scores drawn on the layout."""
    return np.asarray(
        [
            layout.drawn(source, detectors=len(shifts), shifts=shifts, shared=shared, turbine=0.0)
            .areas()
            .max()
            for _ in range(draws)
        ]
    )


def power_rows(layout: Layout, *, draws: int, seed: int, areas: Sequence[float]) -> list[list[str]]:
    """What a family's best must reach, and how often a planted fault's area reaches it."""
    rows: list[list[str]] = []
    source = np.random.default_rng(seed)
    for detectors, shared in ((1, 0.5), (2, 0.5), (2, 0.8), (12, 0.5), (12, 0.8)):
        chance = best_areas(layout, source, draws=draws, shifts=np.zeros(detectors), shared=shared)
        for share in (LEVEL / FAMILIES, LEVEL):
            bar = float(np.quantile(chance, 1.0 - share))
            for carried in ("every", "one")[: 1 if detectors == 1 else 2]:
                powers: list[str] = []
                for area in areas:
                    shifts = np.full(detectors, shift_of(area))
                    if carried == "one":
                        shifts[1:] = 0.0
                    planted = best_areas(
                        layout, source, draws=draws // 10, shifts=shifts, shared=shared
                    )
                    powers.append(f"{(planted > bar).mean():.2f}")
                rows.append(
                    [
                        str(detectors),
                        f"{shared:.1f}",
                        f"{100 * share:.2f} %",
                        f"{bar:.3f}",
                        carried,
                        *powers,
                    ]
                )
    return rows


def size_rows(layout: Layout, *, draws: int, seed: int) -> list[list[str]]:
    """How often the shuffle passes a family of twelve with no fault in it.

    Read where the excerpts of a turbine share a part of their score, which the shuffle of
    excerpts does not know.
    """
    rows: list[list[str]] = []
    source = np.random.default_rng(seed)
    for turbine in (0.0, 0.2, 0.4):
        passed = [
            layout.drawn(
                source, detectors=12, shifts=np.zeros(12), shared=0.5, turbine=turbine
            ).best_beyond_chance(shuffles=999, seed=trial)[1]
            <= LEVEL
            for trial in range(draws)
        ]
        rows.append([f"{turbine:.1f}", f"{100 * float(np.mean(passed)):.2f} %"])
    return rows


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    power = commands.add_parser("power")
    power.add_argument("--root", type=Path, required=True)
    power.add_argument("--draws", type=int, default=20_000)
    power.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    layout = Layout.of(read_events(arguments.root))
    faults = int(layout.anomaly.sum())
    print(
        f"{len(layout.anomaly)} excerpts ranked, {faults} with a fault and "
        f"{len(layout.anomaly) - faults} without, of "
        f"{len(set(zip(layout.farms, layout.turbines, strict=True)))} turbines\n"
    )
    areas = (0.60, 0.65, 0.70, 0.75, 0.80)
    print(
        table(
            [
                "Detectors",
                "Shared variance",
                "Level",
                "Area to reach",
                "Fault carried by",
                *(f"Power at {area:.2f}" for area in areas),
            ],
            power_rows(layout, draws=arguments.draws, seed=arguments.seed, areas=areas),
        )
    )
    print()
    print(
        table(
            ["Variance shared within a turbine", f"Passes at {100 * LEVEL:.1f} % with no fault"],
            size_rows(layout, draws=arguments.draws // 10, seed=arguments.seed),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
