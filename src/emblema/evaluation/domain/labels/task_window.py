from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidTaskWindowError
from emblema.evaluation.domain.identifiers import UnitKey


@dataclass(frozen=True, kw_only=True)
class TaskWindow:
    """One window of a published corpus, as a task addresses it: whose it is, where, and when.

    A task draws labels long before anything reads tokens, so a window is identified here rather
    than carried: the position is the one the published block indexes it under, and the moment it
    ends is what a label is read from.

    Invariants: the position is not negative; the end is finite.

    Attributes:
        unit: Unit the window was cut from.
        position: Index the published block holds the window at.
        ends_at: Last moment the window covers, in the unit's own time axis.
    """

    unit: UnitKey
    position: int
    ends_at: float

    def __post_init__(self) -> None:
        if self.position < 0:
            raise InvalidTaskWindowError(f"window position cannot be negative: {self.position}")
        if not isfinite(self.ends_at):
            raise InvalidTaskWindowError(f"window must end at a finite moment: {self.ends_at}")
