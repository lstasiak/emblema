from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidUnitErrorError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class UnitError:
    """What a candidate got wrong over one unit's windows, as a sum that combines by addition.

    The unit is what a comparison is paired on and what an interval resamples, so the error is
    accumulated here per unit rather than per window: two candidates on the same units are
    compared unit by unit, and a group of units is scored by adding their sums and their counts.

    Invariants: at least one window; the sum is finite and not negative.

    Attributes:
        unit: Unit the windows were cut from.
        squared_error: Sum of squared errors over the unit's windows.
        windows: How many windows the sum runs over.
    """

    unit: UnitKey
    squared_error: float
    windows: int

    def __post_init__(self) -> None:
        if self.windows < 1:
            raise InvalidUnitErrorError(f"a unit's error must cover a window, got {self.windows}")
        if not isfinite(self.squared_error) or self.squared_error < 0.0:
            raise InvalidUnitErrorError(
                f"squared_error must be finite and not negative, got {self.squared_error}"
            )

    @classmethod
    def of(cls, unit: UnitKey, predictions: Iterable[WindowPrediction]) -> Self:
        """The error over ``predictions``, which must all be the unit's.

        Raises:
            InvalidUnitErrorError: If a prediction belongs to another unit, or there is none.
        """
        total, count = 0.0, 0
        for prediction in predictions:
            if prediction.window.unit != unit:
                raise InvalidUnitErrorError(
                    f"a prediction of {prediction.window.unit} was counted against {unit}"
                )
            total += prediction.squared_error
            count += 1
        return cls(unit=unit, squared_error=total, windows=count)

    @property
    def rmse(self) -> float:
        return sqrt(self.squared_error / self.windows)
