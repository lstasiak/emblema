from dataclasses import dataclass
from itertools import pairwise

from emblema.pretraining.domain.exceptions import InvalidSaturationCurveError
from emblema.pretraining.domain.saturation.saturation_point import SaturationPoint

# How far apart in optimiser steps two points of one curve may lie and still be read as runs of
# one compute budget. Epochs are whole, so a share whose units hold more windows than its
# fraction says rounds to a budget a little off the others'; more than this and the curve compares
# budgets, not data.
STEP_TOLERANCE = 0.15


@dataclass(frozen=True)
class SaturationCurve:
    """Validation loss against the share of a corpus trained on, at one compute budget.

    The curve exists to ask one question — does more of this corpus lower the loss of a model
    trained this long — so its points are runs of the same configuration, differing only in the
    share of the training units they read and in how many epochs it took each to spend the same
    optimiser steps. A curve whose points spent different budgets would answer a question about
    compute instead, and is refused.

    Invariants: at least two points; fractions strictly increasing; every point's steps within
    ``STEP_TOLERANCE`` of the largest fraction's.

    Attributes:
        points: The runs, from the smallest share to the largest.
    """

    points: tuple[SaturationPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise InvalidSaturationCurveError("a curve needs at least two shares")
        fractions = [point.fraction for point in self.points]
        if fractions != sorted(set(fractions)):
            raise InvalidSaturationCurveError(f"fractions must increase strictly, got {fractions}")
        reference = self.points[-1].steps
        for point in self.points:
            if abs(point.steps - reference) > STEP_TOLERANCE * reference:
                raise InvalidSaturationCurveError(
                    f"the share {point.fraction:g} took {point.steps} steps against {reference} "
                    f"at the largest share, further apart than {STEP_TOLERANCE:.0%}"
                )

    @property
    def largest(self) -> SaturationPoint:
        """The run over the largest share, which the verdict about the corpus reads."""
        return self.points[-1]

    def gains(self) -> tuple[float, ...]:
        """Relative fall of the validation loss from each share to the next.

        Positive where more data lowered the loss; the last entry is what decides whether the
        corpus still has something to teach at this budget.
        """
        return tuple(
            (before.validation_loss - after.validation_loss) / before.validation_loss
            if before.validation_loss > 0.0
            else 0.0
            for before, after in pairwise(self.points)
        )
