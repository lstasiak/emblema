from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidOutcomeError
from emblema.evaluation.domain.labels.target_kind import TargetKind


@dataclass(frozen=True)
class AnomalyScheme:
    """Which annotated event a window overlaps: zero for none, the event's number otherwise.

    The ground truth answers each window with the number of the event it overlaps, counted from
    one in the order the events are annotated, zero where it overlaps none, and ``EXCLUDED``
    where it overlaps a stretch the task does not score, such as a gap in the downlink. The
    target is the event's number itself rather than a one: every window of one event shares it,
    which is what lets a reading weigh the event once however many windows it spans, and the
    answers stored beside the targets say which event a score was given for. No candidate learns
    the target of a window it answers; the fitting side carries no label, apart from the windows
    of the few events a detector is taught, which travel beside it.

    Attributes:
        NORMAL: The target of a window that overlaps no event.
        EXCLUDED: What the ground truth says of a window the task does not score; never a
            target, since such a window is left out before any label is read.
    """

    NORMAL: ClassVar[float] = 0.0
    EXCLUDED: ClassVar[float] = -1.0

    @property
    def kind(self) -> TargetKind:
        return TargetKind.ANOMALY

    @property
    def scale(self) -> float:
        """The unit the targets are in: an event's number has none, so one."""
        return 1.0

    def scores(self, event: float) -> bool:
        """Whether a window the ground truth answers with ``event`` is scored at all."""
        return event != self.EXCLUDED

    @classmethod
    def is_event(cls, target: float) -> bool:
        """Whether ``target`` is an event's number: a whole number above the normal mark."""
        return isfinite(target) and target > cls.NORMAL and target == int(target)

    def target(self, *, event: float) -> float:
        """The target of a window the ground truth answers with ``event``.

        Raises:
            InvalidOutcomeError: If ``event`` is neither zero nor a whole positive number, the
                excluded mark included: a window the task does not score has no target.
        """
        if not isfinite(event) or event != int(event) or event < self.NORMAL:
            raise InvalidOutcomeError(
                f"an event is numbered from one, with zero for none, got {event}"
            )
        return float(int(event))
