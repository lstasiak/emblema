import math
from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError


@dataclass(frozen=True)
class WindowPlace:
    """Where a published window lies: the time axis it is on, and the span of it the window covers.

    A window with history is joined from the windows before it on the same axis, so a corpus that
    a run reads with history says where each of its windows lies. The axis is the part of the
    corpus the window's unit belongs to, where the corpus arrives in parts, its unit otherwise:
    the units of one part are cut from one stream, a satellite's months from its mission, while a
    unit with no part is a stream of its own.

    Invariants: the axis is non-empty without surrounding whitespace; the span is finite and
    ends after it starts.

    Attributes:
        axis: The time axis the window is on.
        start: Where on it the window starts.
        end: Where on it the window ends.
    """

    axis: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.axis or self.axis != self.axis.strip():
            raise InvalidTrainingCorpusError(
                "a window's axis must be non-empty without surrounding whitespace"
            )
        if not (math.isfinite(self.start) and math.isfinite(self.end)) or self.end <= self.start:
            raise InvalidTrainingCorpusError(
                f"a window must end after it starts, got [{self.start}, {self.end})"
            )
