r"""Whether a detector's scores set the benchmark's faults apart from its quiet excerpts.

An excerpt of the fault benchmark holds one event, the lead-up to a fault or a stretch without
one, so the reading ranks excerpts. An excerpt's statistic is how unusual its event looks beside
its own year. The event is read by the medians of the hourly score over twelve hours running,
twelve hours being the span the benchmark's own counter needs to call a fault found, a median
asking that most of those hours stand high. Its level is the median of those medians: a fault
that grows over days lifts the level through its lead-up, where a single spell lifts one span.
The statistic is one less the share of the blocks of the event's length, slid over the year the
detector was fitted on, whose level is as high.

A block is read through the event's own gaps: only at the hours, counted from its start, where
the event holds a median. An event broken by stops is then set beside blocks broken alike, and
how much of an event a detector could read says nothing of its statistic; read whole against
whole blocks, the events with stops, which are mostly faults, stood low for their gaps alone.
Reading the event against blocks of its own length takes the length out, which nearly spells
the label here, and reading it against its own year takes the turbine's level out. An event no
block reaches ties with every other such.

Only settled hours are read, in the year and in the event alike: an hour whose 24 hours before
all hold a score. An hour after a stop is scored against a shorter and older history, or
answered by a model that does not know the turbine is warming up, and stands higher than a
settled one, so events with stops, which are mostly faults, would stand high for their stops.
An event with no settled span of twelve hours gives an early detector nothing to judge its
turbine by and leaves the ranking, counted.

Beside the verdict's reading: every scored hour read, those after a gap among them; the peak,
the largest of the event's medians; the event's last seven days alone, which makes the events
nearly as long as one another; the unreadable events ranked lowest; and the farms alone whose
status is the operator's.

The verdict is the area under the curve of that statistic between the excerpts with a fault and
those without, pairs taken inside a farm only, for the best detector of a family; its chance is
the same best taken with the labels shuffled inside each farm. The turbines are too few for an
interval that covers as it claims, so the shuffle alone decides and the intervals beside it are
approximate.

Beside the verdict, descriptively: every detector's own area and its approximate interval over
turbines drawn again, and how far its statistic follows the event's length and the share of the
event's rows not read; the answer with no model, the event's length and that share alone, which
a detector must stand above to have read a fault and not a stop or the benchmark's layout;
named detectors' differences, their chance from swapping the two inside each turbine; each
family's best with the excerpt no ranking reads ranked too; and the benchmark's own score at
alarms set on the fitted year, from a half to ten hours in a hundred above, none of them chosen.

    uv run scripts/care_detection_reading.py power --root data/raw/care/CARE_To_Compare \\
        [--draws 20000] [--seed 1]
    uv run scripts/care_detection_reading.py read --root data/raw/care/CARE_To_Compare \\
        --scores DIR --out DIR/read [--pair NAME NAME]... [--trial SEED]

``power`` reads only the benchmark's tables of events: on the layout of the excerpts read it
draws scores with no fault in them and with a planted one, and prints what area a family's best
must reach and how often a true area reaches it. ``read`` joins ``departures-`` and
``behaviour-<farm>-<event>.npz`` from ``--scores`` and writes ``excerpts.csv``, ``areas.csv``,
``verdicts.csv``, ``care.csv`` and, for a pair, ``pair.csv`` under ``--out``. It refuses to read
unless every excerpt that is read has both files, since a family missing from one excerpt
would leave the reading and take its share of the level with it; ``--trial`` reads the excerpts
that are there, shuffles the labels inside each farm before anything is computed, writes its
tables under names of their own without the excerpts' statistics and prints only each table's
size, so the whole reading runs and reads nothing. The benchmark's score counts every excerpt
as the benchmark labels it, the one left out of the rankings among them.
"""

from __future__ import annotations

import argparse
import csv
import enum
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.stats import rankdata, spearmanr

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection  # noqa: E402
from scripts import care_departure_scores, care_normal_behaviour_scores  # noqa: E402
from scripts.care_excerpts import STATUS_UNREAD_WHEN_SCORED, Event, read_events  # noqa: E402
from scripts.care_score import CareReading, CareScore  # noqa: E402
from scripts.reporting import table  # noqa: E402

Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]
Ints = NDArray[np.int64]

# The hours the benchmark's counter needs to call a fault found: 72 rows of ten minutes.
SPAN = 12
# The fewest scored hours of a span whose median is read.
LEAST_SCORED = 9
SHUFFLES = 10_000
# One-sided, over the families a verdict reads, the smaller p-value first.
LEVEL = 0.025
# The families a verdict reads, as the scripts that score them name them.
VERDICTS = (
    care_departure_scores.FAMILY,
    care_normal_behaviour_scores.FAMILY,
    care_normal_behaviour_scores.FLEET,
)
FAMILIES = len(VERDICTS)
# The levels the verdicts' chances are held to, from the smallest chance's to the largest's.
LEVELS = tuple(LEVEL / remaining for remaining in range(FAMILIES, 0, -1))
AVAILABILITY = care_departure_scores.AVAILABILITY
KINDS = ("departures", "behaviour")
LENGTH = "event_length"
UNREAD = "event_unread_share"
# The hours before an hour that must all hold a score for a detector's hour to be read.
SETTLING = 24
# The hours of an event's end that the reading of its last week takes.
LAST_WEEK = 7 * 24
# The shares of the fitted year's scored hours an alarm stands above, for the benchmark's score.
ALARMS = (0.005, 0.01, 0.02, 0.05, 0.10)
RESAMPLES = 2000
SWAPS = 10_000


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


class Reading(enum.Enum):
    """What of an event's running medians is set beside the year's blocks.

    Attributes:
        LEVEL: Their median, which a fault growing through its lead-up lifts.
        PEAK: Their largest, which one spell lifts.
    """

    LEVEL = "level"
    PEAK = "peak"

    def of(self, rows: Floats) -> Floats:
        """Per row, the median or the largest of the values present; missing where none is."""
        if self is Reading.PEAK:
            with np.errstate(invalid="ignore"):
                largest: Floats = np.where(
                    np.isnan(rows).all(axis=1),
                    np.nan,
                    np.where(np.isnan(rows), -np.inf, rows).max(axis=1),
                )
            return largest
        # Sorted, the missing values stand last, so the middle of the values present is found
        # by their count.
        ordered = np.sort(rows, axis=1)
        present = (~np.isnan(rows)).sum(axis=1)
        lower = np.take_along_axis(ordered, np.maximum(present - 1, 0)[:, None] // 2, axis=1)
        upper = np.take_along_axis(ordered, (present // 2)[:, None], axis=1)
        middle: Floats = (lower[:, 0] + upper[:, 0]) / 2.0
        return middle


def unusualness(
    scores: Floats, *, fitted: int, first: int, last: int, reading: Reading = Reading.LEVEL
) -> float | None:
    """How unusual an event's scores are beside the year before it.

    One less the share of the year's blocks of the event's length whose reading is as high as
    the event's, each block read at the hours, from its start, where the event holds a running
    median, and left out where it holds one at fewer than half of them.

    Args:
        scores: The detector's hourly scores over the excerpt, missing where an hour is not
            scored.
        fitted: How many hours, from the first, are the year the detector was fitted on.
        first: The event's first hour.
        last: The event's last hour.
        reading: What of the running medians is compared.

    Returns:
        ``None`` where the event holds no span scored, or the year no block to set beside it.

    Raises:
        ValueError: If the event starts inside the fitted year or lies outside the scores.
    """
    if not fitted <= first <= last < len(scores):
        raise ValueError(
            f"the event's hours {first} to {last} do not follow {fitted} fitted hours of "
            f"{len(scores)}"
        )
    reach = last - first + 1 - SPAN + 1
    event = running_medians(scores[first : last + 1])[SPAN - 1 :]
    held = np.flatnonzero(~np.isnan(event))
    year = running_medians(scores[:fitted])[SPAN - 1 :]
    if not len(held) or reach < 1 or len(year) < reach:
        return None
    blocks = sliding_window_view(year, reach)[:, held]
    alike = blocks[(~np.isnan(blocks)).sum(axis=1) * 2 >= len(held)]
    if not len(alike):
        return None
    of_event = reading.of(event[None, held])[0]
    return 1.0 - float((reading.of(alike) >= of_event).mean())


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

    def strata(self) -> list[Ints]:
        """The excerpts' places, farm by farm."""
        names = np.asarray(self.farms)
        return [np.flatnonzero(names == farm) for farm in dict.fromkeys(self.farms)]

    def pairs(self) -> int:
        """How many pairs of one farm hold a fault and none.

        Raises:
            ValueError: If no farm holds both, so no area is defined.
        """
        pairs = sum(
            int(self.anomaly[stratum].sum()) * int((~self.anomaly[stratum]).sum())
            for stratum in self.strata()
        )
        if not pairs:
            raise ValueError("no farm holds both an excerpt with a fault and one without")
        return pairs

    def areas(self) -> Floats:
        """Per detector, the share of pairs of one farm, a fault and none, ranked rightly."""
        won = np.zeros(len(self.statistics))
        for stratum in self.strata():
            faults = self.anomaly[stratum]
            ranks = rankdata(self.statistics[:, stratum], axis=-1)
            count = int(faults.sum())
            won += ranks[:, faults].sum(axis=1) - count * (count + 1) / 2.0
        areas: Floats = won / self.pairs()
        return areas

    def shuffled_areas(self, shuffles: int, seed: int) -> Floats:
        """``[shuffles, detectors]``: the areas with the labels shuffled inside each farm."""
        draws = np.random.default_rng(seed)
        won = np.zeros((shuffles, len(self.statistics)))
        for stratum in self.strata():
            faults = self.anomaly[stratum]
            ranks = rankdata(self.statistics[:, stratum], axis=-1)
            count = int(faults.sum())
            drawn = draws.permuted(np.tile(faults, (shuffles, 1)), axis=1)
            won += drawn.astype(np.float64) @ ranks.T - count * (count + 1) / 2.0
        shuffled: Floats = won / self.pairs()
        return shuffled

    def best_beyond_chance(self, shuffles: int = SHUFFLES, seed: int = 1) -> tuple[float, float]:
        """The best detector's area, and how often a shuffle's best reaches it."""
        best = float(self.areas().max())
        reached = (self.shuffled_areas(shuffles, seed).max(axis=1) >= best - 1e-12).sum()
        return best, float((1 + reached) / (1 + shuffles))


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
        for share in LEVELS:
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

    Read at each level a verdict's chance can be held to, where the
    excerpts of a turbine share a part of their score, which the shuffle of excerpts does not
    know.
    """
    rows: list[list[str]] = []
    source = np.random.default_rng(seed)
    for turbine in (0.0, 0.2, 0.4):
        chances = np.asarray(
            [
                layout.drawn(
                    source, detectors=12, shifts=np.zeros(12), shared=0.5, turbine=turbine
                ).best_beyond_chance(shuffles=999, seed=trial)[1]
                for trial in range(draws)
            ]
        )
        rows.append(
            [
                f"{turbine:.1f}",
                *(f"{100 * float((chances <= level).mean()):.2f} %" for level in LEVELS),
            ]
        )
    return rows


# --- the reading -----------------------------------------------------------------------------


class View(enum.Enum):
    """How an event is set beside its year.

    Attributes:
        SETTLED: The verdict's: the event's level over its settled hours.
        EVERY_HOUR: Its level over every scored hour, those after a gap among them.
        PEAK: Its peak over its settled hours.
        LAST_WEEK: Its level over the settled hours of its last seven days, which makes the
            events nearly as long as one another.
    """

    SETTLED = "settled"
    EVERY_HOUR = "every_hour"
    PEAK = "peak"
    LAST_WEEK = "last_week"


def settling_of(name: str) -> int:
    """How many hours before an hour must all hold a score for the hour to count as settled.

    Over the fitted years a score in the three hours after a stop of six hours or more stands
    1.1 to 2.6 times a settled hour's, and 13 to 24 hours after it still up to 1.4 times, for
    the departures and the models of normal behaviour alike; later it does not. The answer with
    no model has no history and no such hours.
    """
    return 0 if name == AVAILABILITY else SETTLING


def settled(scores: Floats, before: int) -> Floats:
    """The scores of the hours whose ``before`` hours before all hold a score; the rest missing."""
    if not before:
        return scores
    held = ~np.isnan(scores)
    count = np.concatenate([[0], np.cumsum(held)])
    hours = np.arange(len(scores))
    whole = hours >= before
    whole[before:] &= count[hours[before:]] - count[hours[before:] - before] == before
    return np.where(whole, scores, np.nan)


@dataclass(frozen=True)
class Scored:
    """One excerpt's score files joined.

    Attributes:
        event: The excerpt's event, its label shuffled in a trial.
        series: Each detector's hourly scores, named by family and detector.
        fitted: How many hours, from the first, are the fitted year.
        first: The event's first hour.
        last: Its last hour.
        row_hour: Per row of the scored window, its hour.
        row_read: Per row of the scored window, whether a detector reads it and the benchmark's
            score counts it.
        rows: The positions of the event's first and last rows in the scored window.
        read: Each statistic once it has been asked for; a store, no part of the excerpt.
    """

    event: Event
    series: Mapping[str, Floats]
    fitted: int
    first: int
    last: int
    row_hour: Ints
    row_read: Bools
    rows: tuple[int, int]
    read: dict[tuple[str, View], float | None] = field(default_factory=dict, compare=False)

    @classmethod
    def joined(cls, event: Event, files: Sequence[Mapping[str, NDArray[np.generic]]]) -> Scored:
        """An excerpt from its files, the first of which holds its rows.

        Raises:
            ValueError: If two files place the excerpt's hours differently, or the scored
                window holds no row of the event.
        """
        placed = {(int(file["fitted"]), int(file["first"]), int(file["last"])) for file in files}
        if len(placed) != 1:
            raise ValueError(
                f"the files of excerpt {event.event} of farm {event.farm} place its hours "
                f"differently: {sorted(placed)}"
            )
        ((fitted, first, last),) = placed
        rows = files[0]
        scored = ~rows["row_fitted"].astype(np.bool_)
        ids = rows["row_id"].astype(np.int64)[scored]
        inside = np.flatnonzero((ids >= event.first_row) & (ids <= event.last_row))
        if not len(inside):
            raise ValueError(
                f"the scored window of excerpt {event.event} of farm {event.farm} holds no row "
                "of its event"
            )
        return cls(
            event=event,
            series={
                str(name): file["scores"][index].astype(np.float64)
                for file in files
                for index, name in enumerate(file["arms"].tolist())
            },
            fitted=fitted,
            first=first,
            last=last,
            row_hour=rows["row_hour"][scored].astype(np.int64),
            row_read=rows["row_normal"][scored].astype(np.bool_),
            rows=(int(inside[0]), int(inside[-1])),
        )

    def beside_year(self, name: str, view: View = View.SETTLED) -> float | None:
        """A detector's statistic of the event; ``None`` where it cannot read the event."""
        if (name, view) not in self.read:
            scores = self.series[name]
            if view is not View.EVERY_HOUR:
                scores = settled(scores, settling_of(name))
            first = self.first
            if view is View.LAST_WEEK:
                first = max(first, self.last - LAST_WEEK + 1)
            self.read[name, view] = unusualness(
                scores,
                fitted=self.fitted,
                first=first,
                last=self.last,
                reading=Reading.PEAK if view is View.PEAK else Reading.LEVEL,
            )
        return self.read[name, view]

    def unread_share(self) -> float:
        """The share of the event's rows no detector reads, by their status or for none held."""
        first, last = self.rows
        return 1.0 - float(self.row_read[first : last + 1].mean())

    def statistic(self, name: str, view: View = View.SETTLED) -> float | None:
        """What ranks the excerpt: a detector's statistic, or what stands in for a detector.

        ``LENGTH`` is the event's length and ``UNREAD`` the share of its rows not read; either
        ranks an excerpt with no detector at all.
        """
        if name == LENGTH:
            return float(self.last - self.first + 1)
        if name == UNREAD:
            return self.unread_share()
        return self.beside_year(name, view)

    def flagged(self, name: str, share: float) -> Bools:
        """Per row of the scored window, whether its hour stands above the detector's alarm.

        The alarm is the level ``share`` of the fitted year's scored hours stand above; an hour
        that is not scored is not flagged, and a detector that scored no hour of the year flags
        nothing.
        """
        scores = self.series[name]
        year = scores[: self.fitted]
        year = year[~np.isnan(year)]
        if not len(year):
            return np.zeros(len(self.row_hour), dtype=np.bool_)
        alarm = float(np.quantile(year, 1.0 - share))
        with np.errstate(invalid="ignore"):
            above = scores > alarm
        flagged: Bools = above[self.row_hour]
        return flagged


def shuffled_labels(excerpts: Sequence[Scored], seed: int) -> list[Scored]:
    """The excerpts with their labels shuffled inside each farm, for a run that reads nothing."""
    draws = np.random.default_rng(seed)
    labels = [excerpt.event.anomaly for excerpt in excerpts]
    for farm in dict.fromkeys(excerpt.event.farm for excerpt in excerpts):
        places = [at for at, excerpt in enumerate(excerpts) if excerpt.event.farm == farm]
        for place, label in zip(
            places, draws.permutation([labels[at] for at in places]).tolist(), strict=True
        ):
            labels[place] = bool(label)
    return [
        replace(excerpt, event=replace(excerpt.event, anomaly=label))
        for excerpt, label in zip(excerpts, labels, strict=True)
    ]


@dataclass(frozen=True)
class Family:
    """Detectors' rankings of the excerpts all of them read.

    Attributes:
        names: The detectors.
        excerpts: The excerpts ranked.
        left_out: How many excerpts hold an event one of the detectors cannot read: with no
            settled span of twelve hours, a turbine that kept stopping gives an early detector
            nothing to judge it by.
        ranking: Their ranking.
    """

    names: tuple[str, ...]
    excerpts: tuple[Scored, ...]
    left_out: int
    ranking: Ranking

    @classmethod
    def of(
        cls,
        names: Sequence[str],
        excerpts: Sequence[Scored],
        *,
        view: View = View.SETTLED,
        every: bool = False,
        unread_lowest: bool = False,
        farms: Sequence[str] | None = None,
    ) -> Family:
        """The detectors' ranking of the excerpts.

        Args:
            names: The detectors.
            excerpts: The excerpts read.
            view: How each event is set beside its year.
            every: Whether the excerpts no ranking reads are ranked too.
            unread_lowest: Whether an event a detector cannot read takes the lowest statistic
                instead of leaving the ranking.
            farms: The farms ranked; all where ``None``.
        """
        ranked = [
            excerpt
            for excerpt in excerpts
            if (every or excerpt.event.ranked) and (farms is None or excerpt.event.farm in farms)
        ]
        values = [[excerpt.statistic(name, view) for name in names] for excerpt in ranked]
        kept = [
            at
            for at, row in enumerate(values)
            if unread_lowest or all(value is not None for value in row)
        ]
        return cls(
            names=tuple(names),
            excerpts=tuple(ranked[at] for at in kept),
            left_out=len(ranked) - len(kept),
            ranking=Ranking(
                statistics=np.asarray(
                    [[values[at][column] or 0.0 for at in kept] for column in range(len(names))],
                    dtype=np.float64,
                ),
                anomaly=np.asarray([ranked[at].event.anomaly for at in kept], dtype=np.bool_),
                farms=[ranked[at].event.farm for at in kept],
            ),
        )

    def area_or_none(self, detector: int = 0) -> float:
        """A detector's area; missing where no farm holds both labels among the excerpts."""
        try:
            return float(self.ranking.areas()[detector])
        except ValueError:
            return float("nan")

    def turbines(self) -> list[tuple[str, int]]:
        return [(excerpt.event.farm, excerpt.event.turbine) for excerpt in self.excerpts]

    def intervals(self, resamples: int = RESAMPLES, seed: int = 1) -> Floats:
        """``[detectors, 2]``: each area's 2.5th and 97.5th percentile over turbines drawn again.

        The turbines of each farm are drawn with replacement, a turbine bringing all its
        excerpts. With three to eleven turbines a farm the interval covers less than it says.
        """
        draws = np.random.default_rng(seed)
        keys = self.turbines()
        by_farm: dict[str, list[tuple[str, int]]] = {}
        for key in dict.fromkeys(keys):
            by_farm.setdefault(key[0], []).append(key)
        places = {key: [at for at, own in enumerate(keys) if own == key] for key in set(keys)}
        areas: list[Floats] = []
        for _ in range(resamples):
            chosen = [
                at
                for turbines in by_farm.values()
                for pick in draws.integers(0, len(turbines), size=len(turbines))
                for at in places[turbines[pick]]
            ]
            drawn = Ranking(
                self.ranking.statistics[:, chosen],
                self.ranking.anomaly[chosen],
                [self.ranking.farms[at] for at in chosen],
            )
            # A draw whose every farm holds one label has no pair to rank and is left out.
            if any(0 < drawn.anomaly[stratum].sum() < len(stratum) for stratum in drawn.strata()):
                areas.append(drawn.areas())
        if not areas:
            return np.full((len(self.names), 2), np.nan)
        interval: Floats = np.percentile(np.asarray(areas), [2.5, 97.5], axis=0).T
        return interval

    def difference(self, swaps: int = SWAPS, seed: int = 1) -> tuple[float, float]:
        """The first detector's area less the second's, and its two-sided chance.

        The chance swaps the two detectors' statistics in all of a turbine's excerpts at once,
        each turbine by a coin: were the two alike, a swap would change nothing.
        """
        draws = np.random.default_rng(seed)
        keys = self.turbines()
        index = np.unique([f"{farm}-{turbine}" for farm, turbine in keys], return_inverse=True)[1]
        pair = self.ranking.statistics[:2]
        observed = float(np.subtract(*replace(self.ranking, statistics=pair).areas()))
        reached = 0
        for _ in range(swaps):
            swapped = draws.random(index.max() + 1)[index] < 0.5
            both = np.where(swapped, pair[::-1], pair)
            drawn = float(np.subtract(*replace(self.ranking, statistics=both).areas()))
            reached += abs(drawn) >= abs(observed) - 1e-12
        return observed, float((1 + reached) / (1 + swaps))

    def follows(self, other: str, detector: int = 0) -> float:
        """The rank correlation of a detector's statistic with what stands in for a detector."""
        beside = [excerpt.statistic(other) or 0.0 for excerpt in self.excerpts]
        if len(set(beside)) < 2 or len(set(self.ranking.statistics[detector].tolist())) < 2:
            return float("nan")
        return float(spearmanr(self.ranking.statistics[detector], beside).statistic)


def families_of(excerpts: Sequence[Scored]) -> dict[str, list[str]]:
    """The detectors every excerpt's files hold, by family.

    Raises:
        ValueError: If there is no excerpt.
    """
    if not excerpts:
        raise ValueError("no excerpt's score files were found")
    shared = set.intersection(*(set(excerpt.series) for excerpt in excerpts))
    families: dict[str, list[str]] = {}
    for name in sorted(shared):
        families.setdefault(name.split("/")[0], []).append(name)
    return families


@dataclass(frozen=True)
class DetectorArea:
    """One detector's own ranking of the excerpts it reads.

    Attributes:
        detector: The detector, or what stands in for one.
        excerpts: How many excerpts it ranks.
        left_out: How many hold an event it cannot read.
        beyond_the_year: How many hold an event no block of the fitted year reaches; they tie.
        area: Its area under the curve.
        low: The interval's lower end over turbines drawn again; approximate.
        high: Its upper end.
        own_chance: How often shuffled labels reach the area, the detector taken alone.
        follows_length: The rank correlation of its statistic with the event's length.
        follows_unread: The same with the share of the event's rows not read.
        operators_status: The area in the farms whose status is the operator's.
        by_farm: The area inside each farm.
    """

    detector: str
    excerpts: int
    left_out: int
    beyond_the_year: int
    area: float
    low: float
    high: float
    own_chance: float
    follows_length: float
    follows_unread: float
    operators_status: float
    by_farm: Mapping[str, float]


@dataclass(frozen=True)
class FamilyVerdict:
    """A family's best detector against the best of shuffled labels, and that detector beside.

    Attributes:
        family: The family.
        detectors: How many detectors it holds.
        excerpts: How many excerpts all of them read.
        left_out: How many excerpts hold an event one of them cannot read.
        best: The detector with the largest area.
        best_area: That area.
        chance: How often the best area of shuffled labels reaches it.
        confirmed: Whether the family passes, each verdict's chance held to its level;
            ``None`` for a family no verdict reads.
        operators_status: The best detector's area in the farms whose status is the operator's.
        every_hour: Its area with the hours after a gap read too.
        peak: Its area by the events' peaks.
        last_week: Its area by the events' last seven days.
        unread_lowest: Its area with the events it cannot read ranked lowest.
        with_every_excerpt: Its area with the excerpts no ranking reads ranked too.
    """

    family: str
    detectors: int
    excerpts: int
    left_out: int
    best: str
    best_area: float
    chance: float
    confirmed: bool | None
    operators_status: float
    every_hour: float
    peak: float
    last_week: float
    unread_lowest: float
    with_every_excerpt: float


@dataclass(frozen=True)
class DetectorCare:
    """The benchmark's score of one detector at one alarm."""

    detector: str
    alarm_share: float
    reading: CareReading


def operators_farms(excerpts: Sequence[Scored]) -> list[str]:
    """The farms whose status is the operator's, so that it says what the turbine did."""
    return [
        farm
        for farm in dict.fromkeys(excerpt.event.farm for excerpt in excerpts)
        if farm not in STATUS_UNREAD_WHEN_SCORED
    ]


def area_rows(excerpts: Sequence[Scored], shuffles: int, seed: int) -> list[DetectorArea]:
    """Every detector's own area, its farms' areas, its approximate interval and own chance."""
    rows: list[DetectorArea] = []
    names = [name for family in families_of(excerpts).values() for name in family]
    for name in (*names, LENGTH, UNREAD):
        family = Family.of([name], excerpts)
        (low, high) = family.intervals(seed=seed)[0].tolist()
        rows.append(
            DetectorArea(
                detector=name,
                excerpts=len(family.excerpts),
                left_out=family.left_out,
                beyond_the_year=int((family.ranking.statistics[0] == 1.0).sum()),
                area=family.area_or_none(),
                low=low,
                high=high,
                own_chance=family.ranking.best_beyond_chance(shuffles, seed)[1],
                follows_length=family.follows(LENGTH),
                follows_unread=family.follows(UNREAD),
                operators_status=Family.of(
                    [name], excerpts, farms=operators_farms(excerpts)
                ).area_or_none(),
                by_farm={
                    farm: Family.of([name], excerpts, farms=[farm]).area_or_none()
                    for farm in dict.fromkeys(excerpt.event.farm for excerpt in excerpts)
                },
            )
        )
    return rows


def verdict_rows(excerpts: Sequence[Scored], shuffles: int, seed: int) -> list[FamilyVerdict]:
    """Each family's best area and its chance; the verdicts' passes under the stepped levels.

    A detector that is its own family, as the answer with no model is, stands in the table of
    areas alone.
    """
    read: list[FamilyVerdict] = []
    operators = operators_farms(excerpts)
    for name, detectors in families_of(excerpts).items():
        if len(detectors) < 2 and name not in VERDICTS:
            continue
        family = Family.of(detectors, excerpts)
        areas = family.ranking.areas()
        best, chance = family.ranking.best_beyond_chance(shuffles, seed)
        leader = [detectors[int(areas.argmax())]]
        read.append(
            FamilyVerdict(
                family=name,
                detectors=len(detectors),
                excerpts=len(family.excerpts),
                left_out=family.left_out,
                best=leader[0],
                best_area=best,
                chance=chance,
                confirmed=None,
                operators_status=Family.of(leader, excerpts, farms=operators).area_or_none(),
                every_hour=Family.of(leader, excerpts, view=View.EVERY_HOUR).area_or_none(),
                peak=Family.of(leader, excerpts, view=View.PEAK).area_or_none(),
                last_week=Family.of(leader, excerpts, view=View.LAST_WEEK).area_or_none(),
                unread_lowest=Family.of(leader, excerpts, unread_lowest=True).area_or_none(),
                with_every_excerpt=Family.of(leader, excerpts, every=True).area_or_none(),
            )
        )
    passed = passes({row.family: row.chance for row in read if row.family in VERDICTS})
    return [replace(row, confirmed=passed.get(row.family)) for row in read]


def passes(chances: Mapping[str, float]) -> dict[str, bool]:
    """Which verdicts pass, the smallest chance held to the level over the verdicts' count.

    A verdict's family that is not there enters with a chance of one, so the others keep the
    levels the whole set of verdicts gives them.
    """
    names = list(chances)
    rejected = HolmCorrection(alpha=LEVEL).rejected(
        [chances[name] for name in names], family_size=FAMILIES
    )
    return dict(zip(names, rejected, strict=True))


def care_rows(excerpts: Sequence[Scored]) -> list[DetectorCare]:
    """The benchmark's score of every detector at every alarm."""
    scorer = CareScore()
    rows: list[DetectorCare] = []
    for family, names in families_of(excerpts).items():
        # The answer with no model counts rows the score leaves out, so it has no score.
        if family == AVAILABILITY:
            continue
        for name in names:
            for share in ALARMS:
                reading = scorer.of(
                    [
                        scorer.excerpt(
                            event=excerpt.event.event,
                            anomaly=excerpt.event.anomaly,
                            flagged=excerpt.flagged(name, share),
                            normal=excerpt.row_read,
                            first=excerpt.rows[0],
                            last=excerpt.rows[1],
                        )
                        for excerpt in excerpts
                    ]
                )
                rows.append(DetectorCare(detector=name, alarm_share=share, reading=reading))
    return rows


def flat(row: DetectorArea | FamilyVerdict | DetectorCare) -> dict[str, object]:
    """A row of a table as its file's columns: nested parts spread, numbers to four places."""
    columns: dict[str, object] = {}
    for key, value in asdict(row).items():
        if isinstance(value, dict):
            prefix = "area_" if key == "by_farm" else ""
            columns |= {f"{prefix}{inner}": value[inner] for inner in value}
        else:
            columns[key] = value
    return {
        key: round(value, 5 if "chance" in key else 4) if isinstance(value, float) else value
        for key, value in columns.items()
    }


def excerpt_rows(excerpts: Sequence[Scored]) -> list[dict[str, object]]:
    """Every excerpt's statistic by every detector: what the tables are made from."""
    names = [name for family in families_of(excerpts).values() for name in family]
    rows: list[dict[str, object]] = []
    for excerpt in excerpts:
        row: dict[str, object] = {
            "farm": excerpt.event.farm,
            "event": excerpt.event.event,
            "turbine": excerpt.event.turbine,
            "anomaly": excerpt.event.anomaly,
            "ranked": excerpt.event.ranked,
            "event_hours": excerpt.last - excerpt.first + 1,
            "unread_share": round(excerpt.unread_share(), 4),
        }
        for name in names:
            value = excerpt.statistic(name)
            row[name] = "" if value is None else value
        rows.append(row)
    return rows


def missing_files(events: Sequence[Event], scores: Path) -> list[Path]:
    """The score files a whole reading needs and ``scores`` does not hold."""
    return [
        path
        for event in events
        for kind in KINDS
        if not (path := scores / f"{kind}-{event.farm}-{event.event}.npz").exists()
    ]


def loaded(root: Path, scores: Path, *, partial: bool) -> list[Scored]:  # pragma: no cover
    """The excerpts read with their score files, the departures' first.

    Raises:
        SystemExit: If a file is missing and the reading is not a partial one.
    """
    events = read_events(root)
    if missing := missing_files(events, scores):
        if not partial:
            raise SystemExit(
                f"{len(missing)} score files are missing, the first {missing[0].name}; a "
                "reading takes every excerpt's"
            )
        lacking = {path.name.split("-", 1)[1] for path in missing}
        events = [event for event in events if f"{event.farm}-{event.event}.npz" not in lacking]
    return [
        Scored.joined(
            event,
            [dict(np.load(scores / f"{kind}-{event.farm}-{event.event}.npz")) for kind in KINDS],
        )
        for event in events
    ]


def written(path: Path, rows: Sequence[Mapping[str, object]]) -> None:  # pragma: no cover
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def printed(title: str, rows: Sequence[Mapping[str, object]]) -> None:  # pragma: no cover
    fields = list(dict.fromkeys(key for row in rows for key in row))
    print(f"\n**{title}**\n")
    print(table(fields, [[str(row.get(field, "")) for field in fields] for row in rows]))


def read(arguments: argparse.Namespace) -> None:  # pragma: no cover - reads files
    trial = arguments.trial is not None
    excerpts = loaded(arguments.root, arguments.scores, partial=trial)
    if trial:
        excerpts = shuffled_labels(excerpts, arguments.trial)
        print(f"TRIAL: labels shuffled inside each farm (seed {arguments.trial}); nothing is read")
    faults = sum(excerpt.event.anomaly for excerpt in excerpts)
    print(f"{len(excerpts)} excerpts, {faults} with a fault and {len(excerpts) - faults} without")
    arguments.out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, object]]] = {
        "verdicts": [
            flat(row) for row in verdict_rows(excerpts, arguments.shuffles, arguments.seed)
        ],
        "areas": [flat(row) for row in area_rows(excerpts, arguments.shuffles, arguments.seed)],
        "care": [flat(row) for row in care_rows(excerpts)],
    }
    for names in arguments.pair:
        pair = Family.of(names, excerpts)
        difference, chance = pair.difference(seed=arguments.seed)
        tables.setdefault("pair", []).append(
            {
                "first": names[0],
                "second": names[1],
                "excerpts": len(pair.excerpts),
                "left_out": pair.left_out,
                "difference": round(difference, 4),
                "chance": round(chance, 5),
            }
        )
    if not trial:
        # The excerpts' statistics are real whatever the labels, so a trial does not write them.
        tables["excerpts"] = excerpt_rows(excerpts)
    prefix = f"trial-{arguments.trial}-" if trial else ""
    for name, rows in tables.items():
        written(arguments.out / f"{prefix}{name}.csv", rows)
        if trial:
            print(f"{name}: {len(rows)} rows, columns {', '.join(rows[0])}")
        elif name != "excerpts":
            printed(name, rows)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    power = commands.add_parser("power")
    power.add_argument("--root", type=Path, required=True)
    power.add_argument("--draws", type=int, default=20_000)
    power.add_argument("--seed", type=int, default=1)
    reading = commands.add_parser("read")
    reading.add_argument("--root", type=Path, required=True)
    reading.add_argument("--scores", type=Path, required=True)
    reading.add_argument("--out", type=Path, required=True)
    reading.add_argument("--pair", nargs=2, action="append", default=[], metavar="NAME")
    reading.add_argument("--shuffles", type=int, default=SHUFFLES)
    reading.add_argument("--seed", type=int, default=1)
    reading.add_argument("--trial", type=int, default=None, metavar="SEED")
    arguments = parser.parse_args(argv)
    if arguments.command == "read":
        read(arguments)
        return
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
            [
                "Variance shared within a turbine",
                *(f"Passes at {100 * level:.2f} % with no fault" for level in LEVELS),
            ],
            size_rows(layout, draws=arguments.draws // 10, seed=arguments.seed),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
