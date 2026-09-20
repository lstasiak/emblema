from dataclasses import dataclass
from math import ceil, isfinite

from emblema.pretraining.domain.exceptions import InvalidTrainingBudgetError
from emblema.shared.kernel.learning_rate_schedule import LearningRateSchedule


@dataclass(frozen=True, kw_only=True)
class TrainingBudget:
    """How long a run trains, in how large a step, and from which seed.

    The batch a gradient is taken over is ``batch_size * accumulation_steps``: accumulation buys
    the larger batch on a device that cannot hold it, and is therefore a budget parameter rather
    than a property of the model. Epochs, not steps, are the unit here, because the corpus decides
    how many batches an epoch holds and the same budget has to describe a run over a tenth of it.

    Invariants: every count is positive; the peak rate is positive and finite; the warmup is not
    negative and leaves something to decay over, since a run that only warms up never reaches the
    floor it states; the floor lies in ``[0, 1]``.

    Attributes:
        epochs: Passes over the training windows.
        batch_size: Windows per micro-batch, which is what a device has to hold.
        accumulation_steps: Micro-batches summed into one optimiser step.
        learning_rate: Peak rate, which the schedule scales.
        warmup_epochs: Epochs the rate climbs to the peak over; a fraction of one where a short
            run warms up over part of its first epoch.
        final_lr_fraction: Fraction of the peak the decay ends at.
        seed: Seed of everything the run draws: weights, order of windows, masks.
    """

    epochs: int
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    warmup_epochs: float
    final_lr_fraction: float
    seed: int

    def __post_init__(self) -> None:
        for label, count in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("accumulation_steps", self.accumulation_steps),
        ):
            if count < 1:
                raise InvalidTrainingBudgetError(f"{label} must be positive, got {count}")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise InvalidTrainingBudgetError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        if not isfinite(self.warmup_epochs) or not 0 <= self.warmup_epochs < self.epochs:
            raise InvalidTrainingBudgetError(
                f"warmup_epochs must lie in [0, epochs) so that an epoch is left to decay over, "
                f"got {self.warmup_epochs} of {self.epochs}"
            )
        if not 0.0 <= self.final_lr_fraction <= 1.0:
            raise InvalidTrainingBudgetError(
                f"final_lr_fraction must lie in [0, 1], got {self.final_lr_fraction}"
            )

    @property
    def effective_batch_size(self) -> int:
        """Windows one gradient is taken over."""
        return self.batch_size * self.accumulation_steps

    def steps_per_epoch(self, batches: int) -> int:
        """Optimiser steps an epoch of ``batches`` micro-batches takes.

        A trailing group of fewer micro-batches than the accumulation asks for still steps, so
        the last windows of an epoch reach the weights.

        Raises:
            InvalidTrainingBudgetError: If ``batches`` is not positive.
        """
        if batches < 1:
            raise InvalidTrainingBudgetError(f"an epoch must hold a batch, got {batches}")
        return ceil(batches / self.accumulation_steps)

    def takes_a_step_at(self, index: int, batches: int) -> bool:
        """Whether the micro-batch at ``index`` of an epoch of ``batches`` applies the gradient.

        A group of ``accumulation_steps`` micro-batches makes one step, and the last micro-batch of
        an epoch closes whatever group it is in, so the windows at the end of an epoch reach the
        weights. Keyed on the index rather than on a counter, so a run resumed inside an epoch
        groups its micro-batches exactly as the uninterrupted run did.

        Raises:
            InvalidTrainingBudgetError: If the index is not one of an epoch of that many batches.
        """
        if batches < 1:
            raise InvalidTrainingBudgetError(f"an epoch must hold a batch, got {batches}")
        if not 0 <= index < batches:
            raise InvalidTrainingBudgetError(f"index must lie in [0, {batches}), got {index}")
        return (index + 1) % self.accumulation_steps == 0 or index + 1 == batches

    def schedule(self, batches: int) -> LearningRateSchedule:
        """The rate over the whole run, in optimiser steps, for an epoch of ``batches``.

        Raises:
            InvalidTrainingBudgetError: If ``batches`` is not positive.
        """
        steps = self.steps_per_epoch(batches)
        return LearningRateSchedule(
            warmup_steps=round(self.warmup_epochs * steps),
            total_steps=self.epochs * steps,
            final_fraction=self.final_lr_fraction,
        )
