from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import stdev
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidErrorOverRepeatsError


@dataclass(frozen=True, kw_only=True)
class ErrorOverRepeats:
    """What one side of a comparison scored over its repeats: the pooled error and its spread.

    The pooled error is the one the comparison is made on: every repeat's squared errors added
    before the root. The spread is the standard deviation of the repeats' own errors, reported
    beside the interval and never folded into it: a handful of repeats is not an interval, but it
    is a component of the uncertainty a reader should see.

    Invariants: at least one repeat; the pooled error and the spread are finite and not
    negative; a single repeat has no spread.

    Attributes:
        pooled: The side's error over every repeat at once.
        spread: The standard deviation of the error over the repeats; zero for one repeat.
        repeats: How many repeats the side ran.
    """

    pooled: float
    spread: float
    repeats: int

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise InvalidErrorOverRepeatsError(f"an error needs a repeat, got {self.repeats}")
        for label, value in (("pooled", self.pooled), ("spread", self.spread)):
            if not isfinite(value) or value < 0.0:
                raise InvalidErrorOverRepeatsError(
                    f"{label} must be finite and not negative, got {value}"
                )
        if self.repeats == 1 and self.spread != 0.0:
            raise InvalidErrorOverRepeatsError("a single repeat has no spread")

    @classmethod
    def of(cls, pooled: float, per_repeat: Sequence[float]) -> Self:
        """The error ``pooled`` over the repeats that scored ``per_repeat`` each.

        Raises:
            InvalidErrorOverRepeatsError: If there is no repeat, or a figure is not a finite,
                non-negative number.
        """
        if not per_repeat:
            raise InvalidErrorOverRepeatsError("an error over repeats needs a repeat")
        for value in per_repeat:
            if not isfinite(value) or value < 0.0:
                raise InvalidErrorOverRepeatsError(
                    f"a repeat's error must be finite and not negative, got {value}"
                )
        return cls(
            pooled=pooled,
            spread=stdev(per_repeat) if len(per_repeat) > 1 else 0.0,
            repeats=len(per_repeat),
        )

    def __str__(self) -> str:
        if self.repeats == 1:
            return f"{self.pooled:.3g} over one repeat"
        return f"{self.pooled:.3g} (spread {self.spread:.2g} over {self.repeats} repeats)"
