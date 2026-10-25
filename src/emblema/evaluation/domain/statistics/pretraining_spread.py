from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidPretrainingSpreadError
from emblema.evaluation.domain.statistics.paired_reading import PairedReading


@dataclass(frozen=True, kw_only=True)
class PretrainingSpread:
    """The spread a difference between two pretrainings takes from pretraining alone.

    Read from pairs of pretrainings that differ by their seed alone, each compared by the same
    reader: the squared mean difference less its noise, averaged over the pairs, floored at zero,
    then its root. What is left once the draws of labels and of units are taken out is what the
    pretraining's own seed moves. A floor keeps a quiet set of pairs from making a threshold
    that a single pretraining could cross.

    Invariants: both figures are finite and not negative.

    Attributes:
        own: The spread the pairs show.
        floor: The least spread a rule reads with.
    """

    own: float
    floor: float

    def __post_init__(self) -> None:
        for name, figure in (("own spread", self.own), ("floor", self.floor)):
            if not isfinite(figure) or figure < 0.0:
                raise InvalidPretrainingSpreadError(
                    f"the {name} must be finite and not negative, got {figure}"
                )

    @classmethod
    def of(cls, pairs: Sequence[PairedReading], *, floor: float) -> Self:
        """The spread read from ``pairs``, each a difference between two pretrainings.

        Raises:
            InvalidPretrainingSpreadError: If no pair is given, or the floor is out of range.
        """
        if not pairs:
            raise InvalidPretrainingSpreadError("a spread needs at least one pair of pretrainings")
        excess = sum(pair.mean**2 - pair.noise for pair in pairs) / len(pairs)
        return cls(own=sqrt(max(excess, 0.0)), floor=floor)

    @property
    def value(self) -> float:
        """The spread a rule reads with, the larger of the pairs' and the floor."""
        return max(self.own, self.floor)
