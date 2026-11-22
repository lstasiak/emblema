"""The fault benchmark's own score of early detection, as its publisher's code computes it.

CARE to Compare scores a detector's flags over the scored window of every excerpt by four parts
(Gück and others, *Data* 9(12):138, 2024): the coverage, the F0.5 of the flags over the rows of
an excerpt with a fault, a row counting as faulty inside the annotated lead-up; the accuracy,
the share of an excerpt without a fault left unflagged; the reliability, the F0.5 over excerpts
of whether a fault was found, an excerpt being found once its criticality reaches 72, a counter
that rises with a flagged row and eases with an unflagged one; and the earliness, the share of
the lead-up flagged, its second half weighing less and less towards the fault. The score is
their mean with the accuracy counted twice; it is the accuracy alone where that is a half or
less, and nought where no excerpt was found.

Only rows of normal operation are scored: a row of another status is neither right nor wrong and
leaves the criticality where it stands. The earliness alone reads every row of the lead-up. The
scope of each part follows the publisher's code (``AEFDI/EnergyFaultDetector``,
``evaluation/care_score.py``), which the papers reporting the score ran; the earliness descends
from half the lead-up, as the paper states, where that code's default descends from a quarter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

Bools = NDArray[np.bool_]
Ints = NDArray[np.int64]


def criticality(flagged: Bools, normal: Bools) -> Ints:
    """The counter after each row.

    Up one on a flagged row of normal operation, down one, not below nought, on an unflagged
    one, unchanged on a row of another status.
    """
    counter = np.zeros(len(flagged), dtype=np.int64)
    level = 0
    for row, (flag, scored) in enumerate(zip(flagged.tolist(), normal.tolist(), strict=True)):
        if scored:
            level = level + 1 if flag else max(level - 1, 0)
        counter[row] = level
    return counter


def f_score(*, found: int, missed: int, false: int, beta: float) -> float:
    """The F-score of a count of hits, misses and false alarms; nought where nothing was hit."""
    weight = beta * beta
    below = (1.0 + weight) * found + weight * missed + false
    return (1.0 + weight) * found / below if below else 0.0


@dataclass(frozen=True)
class ScoredExcerpt:
    """What one excerpt's flags come to.

    Attributes:
        event: The excerpt's number in its farm.
        anomaly: Whether the excerpt holds a fault.
        coverage: The F-score over the scored window's normal rows; missing without a fault.
        earliness: The weighted share of the lead-up flagged; missing without a fault.
        accuracy: The share of the scored window's normal rows the flags get right.
        most_critical: The highest criticality from the scored window's start to the event's end.
    """

    event: int
    anomaly: bool
    coverage: float
    earliness: float
    accuracy: float
    most_critical: int


@dataclass(frozen=True)
class CareReading:
    """The score over a set of excerpts and its four parts."""

    coverage: float
    accuracy: float
    reliability: float
    earliness: float
    found: int
    falsely_found: int
    score: float


@dataclass(frozen=True)
class CareScore:
    """The benchmark's score.

    Attributes:
        descends_from: The share of the lead-up after which a flag's weight falls towards nought.
        found_at: The criticality at which an excerpt counts as found.
        beta: The weight of recall against precision in both F-scores.
    """

    descends_from: Fraction = Fraction(1, 2)
    found_at: int = 72
    beta: float = 0.5

    def excerpt(
        self,
        *,
        event: int,
        anomaly: bool,
        flagged: Bools,
        normal: Bools,
        first: int,
        last: int,
    ) -> ScoredExcerpt:
        """One excerpt's parts, from the flags over its scored window in time order.

        Args:
            event: The excerpt's number.
            anomaly: Whether the excerpt holds a fault.
            flagged: Per row of the scored window, whether the detector flags it.
            normal: Per row, whether the turbine is in normal operation.
            first: The position of the event's first row.
            last: The position of the event's last row.

        Raises:
            ValueError: If the flags and the statuses differ in length, or the event lies outside
                them.
        """
        if len(flagged) != len(normal):
            raise ValueError(f"{len(flagged)} flags were given for {len(normal)} statuses")
        if not 0 <= first <= last < len(flagged):
            raise ValueError(f"the event's rows {first} to {last} lie outside {len(flagged)} rows")
        faulty = np.zeros(len(flagged), dtype=np.bool_)
        if anomaly:
            faulty[first : last + 1] = True
        hit = int((flagged & faulty & normal).sum())
        false = int((flagged & ~faulty & normal).sum())
        missed = int((~flagged & faulty & normal).sum())
        scored = int(normal.sum())
        counter = criticality(flagged[: last + 1], normal[: last + 1])
        return ScoredExcerpt(
            event=event,
            anomaly=anomaly,
            coverage=(
                f_score(found=hit, missed=missed, false=false, beta=self.beta)
                if anomaly
                else float("nan")
            ),
            earliness=self.earliness(flagged[first : last + 1]) if anomaly else float("nan"),
            accuracy=(scored - false - missed) / scored if scored else float("nan"),
            most_critical=int(counter.max()),
        )

    def earliness(self, flagged: Bools) -> float:
        """The share of an event's rows flagged, each row weighed by how early it is."""
        weights = self.weights(len(flagged))
        return float((flagged * weights).sum() / weights.sum())

    def weights(self, rows: int) -> NDArray[np.float64]:
        """One until the descent starts, then falling in a line to nought at the last row.

        The arithmetic is the publisher's, on a grid as many times finer as the descent's
        denominator, so the two agree to the last digit.
        """
        scale = self.descends_from.denominator
        start = float(self.descends_from)
        fine = rows * scale
        turn = int(fine * start)
        along = np.linspace(0.0, scale, fine)
        weights = np.full(fine, float(scale))
        weights[turn:] = scale / (1.0 - start) - along[turn:] / (1.0 - start)
        kept: NDArray[np.float64] = weights[::scale] / scale
        return kept

    def of(self, excerpts: Sequence[ScoredExcerpt]) -> CareReading:
        """The score over the excerpts.

        Raises:
            ValueError: If the excerpts do not hold both a fault and none.
        """
        faults = [excerpt for excerpt in excerpts if excerpt.anomaly]
        quiet = [excerpt for excerpt in excerpts if not excerpt.anomaly]
        if not faults or not quiet:
            raise ValueError("the score needs an excerpt with a fault and one without")
        found = sum(excerpt.most_critical >= self.found_at for excerpt in faults)
        falsely = sum(excerpt.most_critical >= self.found_at for excerpt in quiet)
        coverage = float(np.nanmean([excerpt.coverage for excerpt in faults]))
        earliness = float(np.nanmean([excerpt.earliness for excerpt in faults]))
        accuracy = float(np.nanmean([excerpt.accuracy for excerpt in quiet]))
        reliability = f_score(
            found=found, missed=len(faults) - found, false=falsely, beta=self.beta
        )
        if found + falsely == 0:
            score = 0.0
        elif accuracy <= 0.5:
            score = accuracy
        else:
            score = (coverage + earliness + reliability + 2.0 * accuracy) / 5.0
        return CareReading(
            coverage=coverage,
            accuracy=accuracy,
            reliability=reliability,
            earliness=earliness,
            found=found,
            falsely_found=falsely,
            score=score,
        )
