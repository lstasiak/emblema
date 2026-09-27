from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError, InvalidOutcomeError
from emblema.evaluation.domain.labels.target_kind import TargetKind


@dataclass(frozen=True)
class OutcomeScheme:
    """Whether a unit ended in one named outcome: one for the positive outcome, zero otherwise.

    The label belongs to the unit, not to a moment in it, so every window of a unit carries the
    unit's outcome. The outcome is named with the task, because a record often states several
    (death in hospital, length of stay, a severity score) and which one is asked is part of what
    the task is.

    Invariants: the outcome's name is non-blank without surrounding whitespace.

    Attributes:
        outcome: Name of the recorded outcome the label reads.
        OUTCOMES: The two values an outcome takes, the negative one first.
    """

    outcome: str

    OUTCOMES: ClassVar[tuple[float, float]] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.outcome or self.outcome != self.outcome.strip():
            raise InvalidLabelSchemeError(
                f"outcome must be non-blank without surrounding whitespace: {self.outcome!r}"
            )

    @property
    def kind(self) -> TargetKind:
        return TargetKind.BINARY

    @property
    def scale(self) -> float:
        """The unit targets are learnt in: the outcome's own, already in ``{0, 1}``."""
        return 1.0

    def target(self, *, recorded: float) -> float:
        """The target of a window whose unit recorded ``recorded`` for the outcome.

        Raises:
            InvalidOutcomeError: If the record says anything but zero or one.
        """
        if recorded not in self.OUTCOMES:
            raise InvalidOutcomeError(
                f"outcome {self.outcome} must be recorded as 0 or 1, got {recorded}"
            )
        return recorded
