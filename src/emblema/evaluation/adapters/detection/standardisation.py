from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Standardisation:
    """The fitting rows' centre and spread per feature, by which every later row is read.

    Invariants: every feature's spread is at least a tenth of the typical spread, the median
    over every feature, or over the features that have one where more than half have none, so a
    feature that is constant or nearly so on the fitting rows is read over that floor and
    cannot dominate a distance by the few rows that move it; where no feature has a spread,
    the floor is one. A statistic a row cannot support, marked as missing, is put at the fitting
    mean once standardised, so it moves no distance.

    Attributes:
        centre: The fitting mean of every feature.
        scale: The fitting spread of every feature, floored.
    """

    # A feature whose fitting spread lies under this share of the typical one is read over the
    # share instead: with a cut between a spread of nought and one just above it, 39 fitting
    # windows more moved one feature's departures a hundredfold and the farthest distance with
    # them.
    FLOOR_SHARE: ClassVar[float] = 0.1

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fitted_on(cls, rows: NDArray[np.float64]) -> Self:
        with np.errstate(invalid="ignore"):
            centre = np.nan_to_num(np.nanmean(rows, axis=0), nan=0.0)
            spread = np.nan_to_num(np.nanstd(rows, axis=0), nan=0.0)
        return cls(centre=centre, scale=np.maximum(spread, cls.floor_of(spread)))

    @classmethod
    def floor_of(cls, spread: NDArray[np.float64]) -> float:
        """The least spread a feature is read over: a share of the typical spread."""
        typical = float(np.median(spread)) if len(spread) else 0.0
        if typical <= 0.0:
            above = spread[spread > 0.0]
            typical = float(np.median(above)) if len(above) else 1.0 / cls.FLOOR_SHARE
        return cls.FLOOR_SHARE * typical

    def of(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Every row standardised, a missing statistic at the fitting mean."""
        standardised: NDArray[np.float64] = np.nan_to_num(
            (np.asarray(rows, dtype=np.float64) - self.centre) / self.scale, nan=0.0
        )
        return standardised
