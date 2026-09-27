from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidRidgePenaltiesError


@dataclass(frozen=True)
class RidgePenalties:
    """Which strengths of the L2 penalty a closed-form linear head chooses among.

    A ridge fit chooses its penalty by leave-one-out error over the labelled windows, which is
    the choice ridge makes for free; the set it chooses among is stated so that a fit is
    repeatable rather than a function of whatever grid a library ships. Shared by every head
    solved in closed form — the convolution baseline's and the frozen probe's — because the
    grid is a fact about the fit, not about what fed it.

    Invariants: at least one penalty, each positive and finite, none named twice, ascending.

    Attributes:
        values: Strengths of the penalty, ascending.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise InvalidRidgePenaltiesError("a ridge fit chooses among at least one penalty")
        for penalty in self.values:
            if not isfinite(penalty) or penalty <= 0.0:
                raise InvalidRidgePenaltiesError(
                    f"a penalty must be positive and finite, got {penalty}"
                )
        if list(self.values) != sorted(set(self.values)):
            raise InvalidRidgePenaltiesError(
                f"penalties must be ascending and distinct, got {list(self.values)}"
            )

    def __str__(self) -> str:
        """The penalties as one field, for whoever records a fit."""
        return " ".join(f"{penalty:g}" for penalty in self.values)
