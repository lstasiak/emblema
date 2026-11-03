from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RobustScale:
    """A statistic's centre and spread on the fitting rows, by which it is read on every row.

    The median and the scaled median absolute deviation, so a few fitting rows far out, the
    events the side holds unknowingly among them, move neither; two statistics read so share a
    scale and can be averaged. A statistic that varies only by rounding has no spread to read
    over and keeps a spread of one, so it is centred and nothing more.

    Invariants: the spread is positive.

    Attributes:
        centre: The fitting median of the statistic.
        spread: The scaled median absolute deviation about it, or one where there is none.
    """

    # The deviation is put on the scale of a standard deviation, so a spread borrowed from
    # elsewhere reads in the same units.
    DEVIATION_PER_MAD: ClassVar[float] = 1.4826
    # A deviation this small beside its median is the rounding of the values, not a spread:
    # distances expanded as |a|² + |b|² - 2ab carry noise of about 1e-8.
    NO_SPREAD: ClassVar[float] = 1e-9

    centre: float
    spread: float

    @classmethod
    def of(cls, values: NDArray[np.float64]) -> Self:
        centre = float(np.median(values))
        return cls(centre=centre, spread=cls.deviation_of(values) or 1.0)

    @classmethod
    def deviation_of(cls, values: NDArray[np.float64]) -> float:
        """The scaled median absolute deviation of ``values``; nought where it is only rounding."""
        centre = float(np.median(values))
        deviation = cls.DEVIATION_PER_MAD * float(np.median(np.abs(values - centre)))
        return deviation if deviation > cls.NO_SPREAD * (1.0 + abs(centre)) else 0.0

    def read(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Every value less the centre, over the spread."""
        read: NDArray[np.float64] = (values - self.centre) / self.spread
        return read
