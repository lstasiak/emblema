from dataclasses import dataclass
from math import isfinite
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidPracticalFloorError
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats


@dataclass(frozen=True, kw_only=True)
class PracticalFloor:
    """The smallest reduction that counts as a difference in practice, for one budget.

    Two parts, the larger taken: a fixed share of the control's error, which keeps the floor
    from collapsing when a run happens to be quiet, and the spread of that error over the
    control's repeats, which keeps it from sitting below the noise the experiment itself
    generates. A reduction under the floor is reported as distinguishable but practically nil,
    whatever its interval says.

    Invariants: the value is finite and not negative.

    Attributes:
        value: The floor, in the error's unit.
    """

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value < 0.0:
            raise InvalidPracticalFloorError(
                f"a floor must be finite and not negative, got {self.value}"
            )

    @classmethod
    def of(cls, control: ErrorOverRepeats, *, share: float) -> Self:
        """The floor over a control that scored ``control``, its repeats spreading as they did.

        A single repeat has no spread, so the fixed part alone stands.

        Raises:
            InvalidPracticalFloorError: If the share is not a finite, non-negative number.
        """
        if not isfinite(share) or share < 0.0:
            raise InvalidPracticalFloorError(f"share must be finite and not negative, got {share}")
        return cls(value=max(share * control.pooled, control.spread))

    def swallows(self, reduction: float) -> bool:
        """Whether ``reduction`` is too small to matter, whichever way it points."""
        return abs(reduction) < self.value
