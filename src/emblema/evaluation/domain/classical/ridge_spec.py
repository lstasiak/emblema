from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidRidgePenaltiesError, InvalidRidgeSpecError
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties


@dataclass(frozen=True, kw_only=True)
class RidgeSpec:
    """Which penalties a ridge fit chooses among, and on how many threads it solves.

    The penalties are what every closed-form head chooses among (``RidgePenalties``); the
    threads are here for the reason they are in a boosting specification: the linear algebra
    sums in the order the work was split, and two machines agree on an answer only if they agree
    on this.

    Invariants: those of the penalties; at least one thread.

    Attributes:
        penalties: Strengths of the L2 penalty a fit chooses among, ascending.
        threads: How many threads the solve may use, which is part of what it answers.
    """

    penalties: tuple[float, ...]
    threads: int

    def __post_init__(self) -> None:
        try:
            RidgePenalties(self.penalties)
        except InvalidRidgePenaltiesError as error:
            raise InvalidRidgeSpecError(str(error)) from error
        if self.threads < 1:
            raise InvalidRidgeSpecError(f"threads must be positive, got {self.threads}")

    def parameters(self) -> dict[str, str | int]:
        """The knobs flattened to scalars, in a fixed order, for whoever records a fit."""
        return {"ridge_penalties": str(RidgePenalties(self.penalties)), "threads": self.threads}
