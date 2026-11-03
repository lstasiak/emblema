from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Standardisation:
    """The fitting rows' centre and spread per feature, by which every later row is read.

    Invariants: a feature without spread on the fitting rows keeps a spread of one, so it is
    centred and nothing more; a statistic a row cannot support, marked as missing, is put at the
    fitting mean once standardised, so it moves no distance.

    Attributes:
        centre: The fitting mean of every feature.
        scale: The fitting spread of every feature, one where there was none.
    """

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fitted_on(cls, rows: NDArray[np.float64]) -> Self:
        with np.errstate(invalid="ignore"):
            centre = np.nan_to_num(np.nanmean(rows, axis=0), nan=0.0)
            scale = np.nan_to_num(np.nanstd(rows, axis=0), nan=0.0)
        return cls(centre=centre, scale=np.where(scale > 0.0, scale, 1.0))

    def of(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Every row standardised, a missing statistic at the fitting mean."""
        standardised: NDArray[np.float64] = np.nan_to_num(
            (np.asarray(rows, dtype=np.float64) - self.centre) / self.scale, nan=0.0
        )
        return standardised
