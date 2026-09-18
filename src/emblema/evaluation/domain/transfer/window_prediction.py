from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidWindowPredictionError
from emblema.evaluation.domain.labels.task_window import TaskWindow


@dataclass(frozen=True, kw_only=True)
class WindowPrediction:
    """What a candidate said a window's answer was, beside what the label said.

    Kept per window rather than summed, because the summaries a comparison needs are several —
    an error per unit for a paired interval, an error over the last window of each unit, an
    error below the label ceiling — and every one of them is arithmetic over these rows. A run
    that stored only its summary would have to be run again for the next question.

    Invariants: the target and the prediction are finite.

    Attributes:
        window: Which window, and where the block holds it.
        target: What the label scheme said.
        predicted: What the candidate said, in the same unit.
    """

    window: TaskWindow
    target: float
    predicted: float

    def __post_init__(self) -> None:
        for label, value in (("target", self.target), ("predicted", self.predicted)):
            if not isfinite(value):
                raise InvalidWindowPredictionError(f"{label} must be finite, got {value}")

    @property
    def squared_error(self) -> float:
        return (self.predicted - self.target) ** 2
