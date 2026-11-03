from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.exceptions import (
    InvalidAnomalyRankingError,
    InvalidWindowRankingError,
)
from emblema.evaluation.domain.scoring.unit_counts import UnitCounts

# Eight megabytes a table: beyond it, a table of misses per unit costs more than it saves.
_LARGEST_TABLE = 1_000_000


@dataclass(frozen=True, eq=False)
class TiedSweep:
    """The windows of a ranking in order, swept under many draws of its units at once.

    A draw of units only changes how many times each window counts, so an area is a running sum
    along a fixed order. A bootstrap asks for that sum tens of thousands of times over tens of
    thousands of windows, which plain Python loops cannot do in minutes (ADR-0074). Only a group
    of equal answers holding a hit adds to an area, so the hits are kept per group and the misses
    are read at the groups' edges only.

    The hits weigh fractions, so the groups are added one after another in the order of the sweep;
    only within a tied group may the additions fall in another order, at the last bits. The misses
    are whole numbers, whose sums are exact in any order, so they may go through a table of misses
    per unit and edge, or a running count where the units are too many for a table. Either way an
    area does not depend on the machine or its thread count.

    Attributes:
        unit_count: How many units the windows belong to.
        hit_units: The unit of each hit, in sweep order.
        hit_worth: What each hit weighs.
        hit_starts: Where each group holding a hit starts among the hits.
        miss_units: The unit of each miss, in sweep order.
        misses_before: How many misses come before each group holding a hit.
        misses_through: How many misses come up to the end of each group holding a hit.
        largest_table: The most entries a table of misses may hold.
    """

    unit_count: int
    hit_units: NDArray[np.intp]
    hit_worth: NDArray[np.float64]
    hit_starts: NDArray[np.intp]
    miss_units: NDArray[np.intp]
    misses_before: NDArray[np.intp]
    misses_through: NDArray[np.intp]
    largest_table: int

    @classmethod
    def of(
        cls,
        *,
        unit_count: int,
        sequence: Sequence[int],
        hit: Sequence[bool],
        worth: Sequence[float],
        closes_tie: Sequence[bool],
        largest_table: int = _LARGEST_TABLE,
    ) -> Self:
        """The sweep of windows given in order, with the columns a ranking holds."""
        units = np.asarray(sequence, dtype=np.intp)
        hits = np.asarray(hit, dtype=np.bool_)
        closes = np.asarray(closes_tie, dtype=np.bool_)
        group = np.concatenate(([0], np.cumsum(closes[:-1], dtype=np.intp)))
        misses_up_to = np.concatenate(([0], np.cumsum(~hits, dtype=np.intp)))
        hit_groups = group[hits]
        firsts = np.flatnonzero(np.diff(hit_groups, prepend=-1))
        edges = np.flatnonzero(closes)
        ends = edges[hit_groups[firsts]] + 1
        starts = np.concatenate(([0], edges[:-1] + 1))[hit_groups[firsts]]
        return cls(
            unit_count=unit_count,
            hit_units=units[hits],
            hit_worth=np.asarray(worth, dtype=np.float64)[hits],
            hit_starts=firsts,
            miss_units=units[~hits],
            misses_before=misses_up_to[starts],
            misses_through=misses_up_to[ends],
            largest_table=largest_table,
        )

    def average_precisions(self, counts: UnitCounts) -> list[float]:
        """The area under the precision-recall curve per draw, from the highest answer down.

        Raises:
            InvalidAnomalyRankingError: If a draw leaves no event or no normal window.
        """
        gained = self._gained(counts)
        found = np.add.accumulate(gained, axis=1)
        if not (found[:, -1].all() and self._misses_in_all(counts).all()):
            raise InvalidAnomalyRankingError("the weights leave no event or no normal window")
        taken = found + self._misses_through(counts)
        precise = np.divide(gained * found, taken, out=np.zeros_like(taken), where=gained != 0.0)
        areas = np.add.accumulate(precise, axis=1)[:, -1] / found[:, -1]
        return [float(area) for area in areas]

    def aurocs(self, counts: UnitCounts) -> list[float]:
        """The area under the ROC curve per draw, from the lowest answer up, a tie counting half.

        Raises:
            InvalidWindowRankingError: If a draw leaves one outcome only.
        """
        gained = self._gained(counts)
        positives = np.add.accumulate(gained, axis=1)[:, -1]
        negatives = self._misses_in_all(counts)
        if not (positives.all() and negatives.all()):
            raise InvalidWindowRankingError("the weights leave one outcome only")
        if self._fits_a_table:
            below = counts.rows @ self._table_before
            through = counts.rows @ self._table_through
        else:
            running = self._running_misses(counts)
            below = running[:, self.misses_before]
            through = running[:, self.misses_through]
        beside = through - below
        ordered = np.add.accumulate(gained * (below + beside / 2.0), axis=1)[:, -1]
        areas = ordered / (positives * negatives)
        return [float(area) for area in areas]

    def _gained(self, counts: UnitCounts) -> NDArray[np.float64]:
        """What each group holding a hit gains under each draw."""
        gained = counts.rows[:, self.hit_units] * self.hit_worth
        if len(self.hit_starts) == len(self.hit_units):
            return gained
        return np.add.reduceat(gained, self.hit_starts, axis=1)

    def _misses_in_all(self, counts: UnitCounts) -> NDArray[np.float64]:
        per_unit = np.bincount(self.miss_units, minlength=self.unit_count).astype(np.float64)
        return counts.rows @ per_unit

    def _misses_through(self, counts: UnitCounts) -> NDArray[np.float64]:
        if self._fits_a_table:
            return counts.rows @ self._table_through
        return self._running_misses(counts)[:, self.misses_through]

    def _running_misses(self, counts: UnitCounts) -> NDArray[np.float64]:
        """The running count of misses under each draw, with a leading zero."""
        running = np.zeros((len(counts.rows), len(self.miss_units) + 1))
        misses = running[:, 1:]
        np.take(counts.rows, self.miss_units, axis=1, out=misses, mode="clip")
        np.add.accumulate(misses, axis=1, out=misses)
        return running

    @property
    def _fits_a_table(self) -> bool:
        return self.unit_count * len(self.hit_starts) <= self.largest_table

    @cached_property
    def _table_before(self) -> NDArray[np.float64]:
        return self._table_up_to(self.misses_before)

    @cached_property
    def _table_through(self) -> NDArray[np.float64]:
        return self._table_up_to(self.misses_through)

    def _table_up_to(self, edges: NDArray[np.intp]) -> NDArray[np.float64]:
        """How many misses of each unit (row) come before each edge (column)."""
        table = np.zeros((self.unit_count, len(edges)))
        section = np.searchsorted(edges, np.arange(len(self.miss_units)), side="right")
        kept = section < len(edges)
        np.add.at(table, (self.miss_units[kept], section[kept]), 1.0)
        return np.cumsum(table, axis=1)
