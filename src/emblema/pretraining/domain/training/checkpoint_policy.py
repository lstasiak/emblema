from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidCheckpointPolicyError


@dataclass(frozen=True, kw_only=True)
class CheckpointPolicy:
    """How often a run writes the state it could be resumed from.

    Stated in optimiser steps rather than epochs: an epoch at the published scale is hours, and a
    session that drops halfway through one must not lose them. Not in wall-clock time, which would
    make what a run left behind depend on how busy its machine was, and not on the metric, which
    would be choosing a model rather than being able to carry on. Where a checkpoint goes and what
    it holds is the runtime's business; the policy only says when.

    Invariants: the interval is positive.

    Attributes:
        every_steps: Optimiser steps between checkpoints.
    """

    every_steps: int

    def __post_init__(self) -> None:
        if self.every_steps < 1:
            raise InvalidCheckpointPolicyError(
                f"every_steps must be positive, got {self.every_steps}"
            )

    def due_at(self, step: int) -> bool:
        """Whether the run writes a checkpoint after ``step``, counted from one.

        Raises:
            InvalidCheckpointPolicyError: If ``step`` is not positive.
        """
        if step < 1:
            raise InvalidCheckpointPolicyError(f"steps are counted from one, got {step}")
        return step % self.every_steps == 0
