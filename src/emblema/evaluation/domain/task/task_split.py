from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit


@dataclass(frozen=True)
class TaskSplit:
    """Which units of a task tune it, which validate it and which it is finally scored on.

    The split is made on units, never on windows: windows of one unit overlap, so a split that
    cut between them would put a window's neighbour on the other side and measure memory rather
    than generalisation. It is recorded when the task is created and never recomputed, because a
    split derived on demand moves whenever anything it derives from does.

    Invariants: the tuning and validation sides hold at least one unit each, and no unit appears
    on more than one of the three sides.

    Attributes:
        tuning: Units labels are drawn from and models are fitted on.
        validation: Units every number reported before the final run is measured on.
        test: Units held for the final run alone.
    """

    tuning: frozenset[UnitKey]
    validation: frozenset[UnitKey]
    test: FrozenTestSplit

    def __post_init__(self) -> None:
        if not self.tuning:
            raise InvalidTaskSplitError("the tuning side must hold at least one unit")
        if not self.validation:
            raise InvalidTaskSplitError("the validation side must hold at least one unit")
        for left, right, label in (
            (self.tuning, self.validation, "tuning and validation"),
            (self.tuning, self.test.units, "tuning and test"),
            (self.validation, self.test.units, "validation and test"),
        ):
            shared = sorted(str(unit) for unit in left & right)
            if shared:
                raise InvalidTaskSplitError(f"units on both the {label} sides: {shared}")
