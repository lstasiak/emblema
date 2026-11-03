from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.robust_scale import RobustScale
from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError


@dataclass(frozen=True)
class ScoreAlarm:
    """Where a detector's scores raise an alarm, set over the fitting windows' own scores.

    The level is the quantile of the fitting scores that leaves the stated contamination of them
    above it, so a detector answers its score less the level and positive is an alarm; every
    detector answers so, whatever its score is.

    Set with the unit of every fitting score, the alarm reads each score against its own unit
    first, the score less the median of the unit's scores over their spread, and sets the level
    over the scores read so; it then reads only scores that come with their units.

    Attributes:
        level: The score above which a window raises an alarm.
        per_unit: Whether a score is read against the other scores of its unit.
        unit_spread: The spread a unit without one of its own is read over: the fitting side's,
            over all its scores; one where that has none either.
    """

    level: float
    per_unit: bool = False
    unit_spread: float = 1.0

    @classmethod
    def set_over(
        cls,
        scores: NDArray[np.float64],
        *,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The alarm over the fitting windows' ``scores``, each of ``units`` if given.

        Raises:
            ValueError: If ``units`` does not name one unit per score.
            InvalidScoredOutcomeError: If units are given and none holds two scores.
        """
        unit_spread = 1.0
        if units is not None:
            unit_spread = cls._spread(scores) or 1.0
            scores = cls._against_own_unit(scores, units, fallback=unit_spread)
        return cls(
            level=float(np.quantile(scores, 1.0 - contamination)),
            per_unit=units is not None,
            unit_spread=unit_spread,
        )

    def above(
        self, scores: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """Every score less the level, read against its unit first when the alarm reads units.

        The scores of a unit are read together: one's place depends on the others given in the
        same call, so a unit is scored whole, once.

        Raises:
            ValueError: If the scores come with units and the alarm reads none, or the other way
                round, or ``units`` does not name one unit per score.
            InvalidScoredOutcomeError: If units are given and none holds two scores.
        """
        self.require(units)
        if units is not None:
            scores = self._against_own_unit(scores, units, fallback=self.unit_spread)
        return scores - self.level

    def require(self, units: Sequence[str] | None) -> None:
        """Refuse scores read against units the alarm does not read, or without those it does.

        Raises:
            ValueError: If units are given and the alarm reads none, or the other way round.
        """
        if (units is not None) != self.per_unit:
            raise ValueError(
                "this detector reads a score against its unit and was given no units"
                if self.per_unit
                else "this detector reads no unit and was given units"
            )

    @staticmethod
    def _spread(scores: NDArray[np.float64]) -> float:
        """The scaled median absolute deviation of ``scores``; zero where it is only rounding."""
        return RobustScale.deviation_of(scores)

    @classmethod
    def _against_own_unit(
        cls, scores: NDArray[np.float64], units: Sequence[str], *, fallback: float
    ) -> NDArray[np.float64]:
        """Every score less its unit's median, over the unit's scaled median absolute deviation.

        A unit whose scores have no such spread, a lone score or scores alike, is read over
        ``fallback`` instead, so its scores stay on the scale of every other unit's.

        Raises:
            ValueError: If ``units`` does not name one unit per score.
            InvalidScoredOutcomeError: If no unit holds two scores.
        """
        keys = np.asarray([str(unit) for unit in units])
        if len(keys) != len(scores):
            raise ValueError(f"{len(keys)} units were given for {len(scores)} rows")
        found, counts = np.unique(keys, return_counts=True)
        if counts.max() < 2:
            raise InvalidScoredOutcomeError(
                "no unit holds two windows, so no window has its unit to be read against"
            )
        read = np.empty_like(scores)
        for unit in found:
            mask = keys == unit
            own = scores[mask]
            read[mask] = (own - float(np.median(own))) / (cls._spread(own) or fallback)
        return read
