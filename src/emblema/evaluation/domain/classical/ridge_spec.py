from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidRidgeSpecError


@dataclass(frozen=True, kw_only=True)
class RidgeSpec:
    """Which penalties a ridge fit chooses among, and on how many threads it solves.

    The penalty is chosen by leave-one-out error over the labelled windows, which is the choice
    ridge makes for free; the set it chooses among is stated here so that a fit is repeatable
    rather than a function of whatever grid a library ships. The threads are here for the reason
    they are in a boosting specification: the linear algebra sums in the order the work was
    split, and two machines agree on an answer only if they agree on this.

    Invariants: at least one penalty, each positive and finite, none named twice, ascending; at
    least one thread.

    Attributes:
        penalties: Strengths of the L2 penalty a fit chooses among, ascending.
        threads: How many threads the solve may use, which is part of what it answers.
    """

    penalties: tuple[float, ...]
    threads: int

    def __post_init__(self) -> None:
        if not self.penalties:
            raise InvalidRidgeSpecError("a ridge fit chooses among at least one penalty")
        for penalty in self.penalties:
            if not isfinite(penalty) or penalty <= 0.0:
                raise InvalidRidgeSpecError(f"a penalty must be positive and finite, got {penalty}")
        if list(self.penalties) != sorted(set(self.penalties)):
            raise InvalidRidgeSpecError(
                f"penalties must be ascending and distinct, got {list(self.penalties)}"
            )
        if self.threads < 1:
            raise InvalidRidgeSpecError(f"threads must be positive, got {self.threads}")

    def parameters(self) -> dict[str, str | int]:
        """The knobs flattened to scalars, in a fixed order, for whoever records a fit."""
        return {
            "ridge_penalties": " ".join(f"{penalty:g}" for penalty in self.penalties),
            "threads": self.threads,
        }
