from dataclasses import dataclass
from math import isfinite
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidSummaryTermError

# The summaries of a channel's hidden values the term predicts, in the order its head answers them.
SUMMARIES = ("mean", "minimum", "maximum")


@dataclass(frozen=True, kw_only=True)
class SummaryTerm:
    """A second term of the objective: each channel's level and extremes of the hidden part.

    Beside the loss at each hidden token, a run may ask for the mean, the minimum and the maximum
    of every channel's hidden values in a window, one unit per channel and window however many
    tokens it hid, so that a channel measured twice a stay weighs as much as one measured every
    minute and the extremes are asked for, not only the values between them. The term is a
    parameter of the method like the loss and the target, stated by the experiment, reported with
    the run and digested into its signature; a run that states none renders nothing, as every run
    before the term existed.

    Invariants: the weight is finite and not negative; zero is no term.

    Attributes:
        weight: What one channel and window's loss weighs against one hidden token's loss.
    """

    weight: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.weight) or self.weight < 0.0:
            raise InvalidSummaryTermError(
                f"weight must be finite and not negative, got {self.weight}"
            )

    @classmethod
    def none(cls) -> Self:
        """No summaries: the objective every run stated before the term existed."""
        return cls()

    @classmethod
    def of(cls, *, weight: float) -> Self:
        return cls(weight=weight)

    @property
    def is_stated(self) -> bool:
        return self.weight > 0.0
