"""A new turbine's own hours since its start, read by what corrects the farm's trees.

The farm's trees answer a turbine read as new from 47 of its hours and the forecast hour's
operating conditions. By its fourteenth day the turbine has read two weeks, which a forecaster
handed every hour since the start reads and the trees do not. Here the classical side reads
them too: the trees answer every own hour before the forecast hour that they could have
answered as a cell, and what they miss there, the residual, is modelled three ways and added to
their answer of the cell:

- ``_offset``: the residuals' mean, shrunk towards nought as by a day of nought residuals;
- ``_ridge``: a ridge of the residual on the trees' answer, the residual of the hour before and
  the hour's operating conditions, every coefficient shrunk towards no correction, under the
  penalty that would have answered the hours so far best;
- ``_boosted``: small trees fitted to the residuals on the trees' own rows, once a day.

A correction reads the hours before its cell's alone, the small trees those before its cell's
day, and is nought where there are none, which leaves the trees' answer. Only numpy is
imported; ``fleet_tree_answers.py`` hands the trees and writes the answers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.cold_start_cells import CONTEXT, PENALTIES
from scripts.fleet_cold_start_cells import Rows, scorable
from scripts.fleet_hours import HOURS_PER_DAY

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
# A model fitted on rows and targets by a seed, answering rows.
Fit = Callable[[Floats, Floats, int], Callable[[Floats], Floats]]
SUFFIXES = ("_offset", "_ridge", "_boosted")
# The offset is shrunk as if a day of hours with no residual had been read before the first.
PRIOR_HOURS = 24
# Small trees are fitted once the trees have answered two days of the turbine's hours.
LEAST_ROWS = 2 * HOURS_PER_DAY
# From the most shrunk on, so that hours which tell the penalties apart by nothing leave the
# trees' answer as it is.
SHRUNK_FIRST = tuple(sorted(PENALTIES, reverse=True))


@dataclass(frozen=True)
class History:
    """One channel's own hours that the trees answered, in order.

    Attributes:
        hours: The hours.
        rows: The trees' rows of each.
        answers: The trees' answer of each.
        conditions: The hour's operating conditions of each, as the rows hold them.
        residuals: The reading of each less the trees' answer.
    """

    hours: Ints
    rows: Floats
    answers: Floats
    conditions: Floats
    residuals: Floats

    def before(self, hours: Ints) -> Ints:
        """How many of the own hours lie before each of ``hours``."""
        out: Ints = np.searchsorted(self.hours, hours, side="left").astype(np.int64)
        return out

    def previous(self, hours: Ints) -> Floats:
        """The residual of the hour before each of ``hours``, nought where the trees gave none."""
        if not len(self.hours):
            return np.zeros(len(hours))
        at = np.minimum(np.searchsorted(self.hours, hours - 1), len(self.hours) - 1)
        return np.where(self.hours[at] == hours - 1, self.residuals[at], 0.0)


def history_of(
    values: Floats,
    recent: Floats,
    kind: Rows,
    channel: int,
    start: int,
    end: int,
    model: Callable[[Floats], Floats],
) -> History:
    """The channel's hours from ``start`` to before ``end`` that the trees answer.

    ``values`` are the new turbine's, unknown before ``start``, and ``recent`` their
    ``latest``. An hour is read where a cell of it would be scored and its rows are whole.
    """
    first = max(start - CONTEXT, 0)
    scored = scorable(values[first:end])[start - first :, channel]
    hours = start + np.flatnonzero(scored).astype(np.int64)
    rows = kind.of(values, recent, hours, channel) if len(hours) else np.zeros((0, 0))
    kept = np.isfinite(rows).all(axis=1)
    hours, rows = hours[kept], rows[kept]
    answers = model(rows) if len(hours) else np.zeros(0)
    conditions = values[hours][:, list(kind.conditions)]
    return History(hours, rows, answers, conditions, values[hours, channel] - answers)


def offsets(history: History, hours: Ints) -> Floats:
    """The mean of the residuals before each of ``hours``, shrunk by ``PRIOR_HOURS``."""
    sums = np.concatenate([[0.0], np.cumsum(history.residuals)])
    count = history.before(hours)
    out: Floats = sums[count] / (count + PRIOR_HOURS)
    return out


def ridge_rows(answers: Floats, previous: Floats, conditions: Floats) -> Floats:
    return np.column_stack([np.ones(len(answers)), answers, previous, conditions])


def ridged(history: History, hours: Ints, answers: Floats, conditions: Floats) -> Floats:
    """The ridge's residual for each of ``hours``, fitted on the own hours before it.

    Each own hour is first answered by the ridge of the hours before it under every penalty;
    an hour's penalty is the one that has missed those before it least.
    """
    read = ridge_rows(history.answers, history.previous(history.hours), history.conditions)
    width = read.shape[1]
    products = np.einsum("ni,nj->nij", read, read)
    gram = np.concatenate([np.zeros((1, width, width)), np.cumsum(products, axis=0)])
    moment = np.concatenate(
        [np.zeros((1, width)), np.cumsum(read * history.residuals[:, None], axis=0)]
    )
    # ``fitted[p, n]``: the coefficients under penalty p of the first n own hours.
    fitted = np.stack(
        [
            np.linalg.solve(gram + penalty * np.eye(width), moment[..., None])[..., 0]
            for penalty in SHRUNK_FIRST
        ]
    )
    missed = (history.residuals - np.einsum("pnf,nf->pn", fitted[:, :-1], read)) ** 2
    so_far = np.concatenate([np.zeros((len(SHRUNK_FIRST), 1)), np.cumsum(missed, axis=1)], axis=1)
    chosen = so_far.argmin(axis=0)
    count = history.before(hours)
    new = ridge_rows(answers, history.previous(hours), conditions)
    out: Floats = np.einsum("cf,cf->c", fitted[chosen[count], count], new)
    return out


def boosted(
    history: History, hours: Ints, rows: Floats, start: int, small: Fit, seed: int
) -> Floats:
    """Small trees' residual for each of ``hours``, fitted on the own hours before its day."""
    out = np.zeros(len(hours))
    days = (hours - start) // HOURS_PER_DAY
    for day in np.unique(days):
        count = int(history.before(np.asarray([start + day * HOURS_PER_DAY], dtype=np.int64))[0])
        if count < LEAST_ROWS:
            continue
        mine = days == day
        out[mine] = small(history.rows[:count], history.residuals[:count], seed)(rows[mine])
    return out


def corrected(
    history: History,
    hours: Ints,
    rows: Floats,
    answers: Floats,
    conditions: Floats,
    *,
    start: int,
    small: Fit,
    seed: int,
) -> dict[str, Floats]:
    """The trees' ``answers`` for the cells at ``hours``, corrected each way, by suffix."""
    return {
        "_offset": answers + offsets(history, hours),
        "_ridge": answers + ridged(history, hours, answers, conditions),
        "_boosted": answers + boosted(history, hours, rows, start, small, seed),
    }
