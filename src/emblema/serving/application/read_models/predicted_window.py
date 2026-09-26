from dataclasses import dataclass
from math import isfinite

from emblema.serving.domain.exceptions import NonFiniteAnswerError


@dataclass(frozen=True, kw_only=True)
class PredictedWindow:
    """The answer for one window, beside what it was computed over and what was left out.

    Invariants: the answer is a finite number. A model that answers NaN or infinity has failed,
    and a client would otherwise read the failure as a missing value.

    Attributes:
        prediction: The candidate's answer, in the task's unit.
        channels_used: The channels the answer was computed over, sorted.
        channels_ignored: Channels the request had readings on that the model does not know.
        warnings: Every way the answer is less than what was asked, in words.
    """

    prediction: float
    channels_used: tuple[str, ...]
    channels_ignored: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.prediction):
            raise NonFiniteAnswerError(f"the model answered {self.prediction} for a window")
