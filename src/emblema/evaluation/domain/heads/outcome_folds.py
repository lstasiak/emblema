from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidOutcomeFoldsError, UnfoldableOutcomesError
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme


@dataclass(frozen=True)
class OutcomeFolds:
    """Which windows of a sample a penalty is judged on, each by a head fitted without them.

    A head over outcomes fitted by penalised likelihood has no leave-one-out error for free, as
    a least-squares one has, so its penalty is chosen by the log-loss of folds held out in
    turn. The folds keep the sample's proportion of the outcomes: a fold richer in positives
    than the rest would leave a head fitted on fewer of them to answer it.

    Each outcome's windows are dealt to the folds in turn, in the sample's order, the second
    outcome taking up the turn where the first left off, so every fold holds each outcome within
    one window of its share and the folds differ in size by one at most. The order is the one
    the sample was drawn into, so the folds replay with it and need no seed of their own.

    Invariants: at least one window; the folds are numbered from zero and each holds a window.

    Attributes:
        assignment: The fold of each window, in the sample's order.
    """

    # The count cross-validation usually takes: each head is fitted on four fifths of the sample,
    # close to the head it stands in for, at the cost of five fits per penalty.
    COUNT: ClassVar[int] = 5
    # Two windows of an outcome, dealt in turn, land in two folds, so every head fitted without
    # a fold sees both outcomes.
    _FEWEST_OF_EACH: ClassVar[int] = 2

    assignment: tuple[int, ...]

    def __post_init__(self) -> None:
        folds = set(self.assignment)
        if not folds or folds != set(range(len(folds))):
            raise InvalidOutcomeFoldsError(
                f"folds must be numbered from zero, each holding a window, got {sorted(folds)}"
            )

    @classmethod
    def of(cls, outcomes: Sequence[float]) -> Self:
        """The folds of a sample whose windows have ``outcomes``, in the sample's order.

        Raises:
            UnfoldableOutcomesError: If an outcome is not zero or one, or either outcome has
                fewer than two windows, which leaves a fold whose head never sees it.
        """
        if any(outcome not in OutcomeScheme.OUTCOMES for outcome in outcomes):
            raise UnfoldableOutcomesError("every outcome must be zero or one")
        for outcome in OutcomeScheme.OUTCOMES:
            held = sum(1 for value in outcomes if value == outcome)
            if held < cls._FEWEST_OF_EACH:
                raise UnfoldableOutcomesError(
                    f"folds need at least {cls._FEWEST_OF_EACH} windows of each outcome, "
                    f"got {held} of outcome {outcome:g}"
                )
        count = min(cls.COUNT, len(outcomes))
        assignment = [0] * len(outcomes)
        turn = 0
        for outcome in OutcomeScheme.OUTCOMES:
            for place, value in enumerate(outcomes):
                if value == outcome:
                    assignment[place] = turn % count
                    turn += 1
        return cls(assignment=tuple(assignment))

    @property
    def count(self) -> int:
        return max(self.assignment) + 1

    def held_out(self, fold: int) -> tuple[int, ...]:
        """Places of the windows ``fold`` is judged on."""
        return tuple(place for place, owner in enumerate(self.assignment) if owner == fold)

    def kept(self, fold: int) -> tuple[int, ...]:
        """Places of the windows the head judged on ``fold`` is fitted on."""
        return tuple(place for place, owner in enumerate(self.assignment) if owner != fold)
