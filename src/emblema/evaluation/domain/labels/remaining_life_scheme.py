from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError, UnlabelledWindowError


@dataclass(frozen=True)
class RemainingLifeScheme:
    """How much life a window is said to have left: the time to failure, held under a ceiling.

    Degradation is not visible from the first cycle of a healthy unit, so a target that counted
    every remaining cycle would ask the model to predict a number nothing in the window carries.
    Holding early life at a ceiling is the convention this kind of task is reported under, and it
    is fixed with the task rather than per run: a ceiling chosen once the errors are visible is a
    threshold chosen after the numbers.

    Invariants: the ceiling is positive and finite.

    Attributes:
        ceiling: Largest remaining life a window may be labelled with.
    """

    ceiling: float

    def __post_init__(self) -> None:
        if not (isfinite(self.ceiling) and self.ceiling > 0):
            raise InvalidLabelSchemeError(f"ceiling must be positive and finite: {self.ceiling}")

    @property
    def scale(self) -> float:
        """The unit targets are learnt in: the ceiling, so every target lies in ``[0, 1]``."""
        return self.ceiling

    def target(self, *, failed_at: float, ends_at: float) -> float:
        """What a window ending at ``ends_at`` has left of a unit that failed at ``failed_at``.

        A window that ends exactly at the failure is labelled zero and is the most informative
        one a unit has: it is what the moment before a failure looks like.

        Raises:
            UnlabelledWindowError: If the window reaches past the failure, which means it was cut
                from another unit's axis or from a unit that did not run to failure.
        """
        remaining = failed_at - ends_at
        if remaining < 0:
            raise UnlabelledWindowError(
                f"window ending at {ends_at} reaches past the failure at {failed_at}"
            )
        return min(remaining, self.ceiling)
