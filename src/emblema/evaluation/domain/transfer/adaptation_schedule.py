from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidAdaptationScheduleError
from emblema.shared.kernel.learning_rate_schedule import LearningRateSchedule


@dataclass(frozen=True, kw_only=True)
class AdaptationSchedule:
    """How long a task is learnt from its labels, in how large a step, and under what decay.

    A fixed number of epochs, scored after the last: stopping when the validation error turns
    would spend the validation side on a decision per run, and every number reported on it
    afterwards would be a little too good. The weight decay is stated rather than inherited,
    because a mode that caps the degrees of freedom is compared against modes that regularise
    otherwise, and what those did has to be on the record. The rate's shape is stated for the
    same reason: a fresh encoder that never leaves the trivial predictor under a constant rate
    is a schedule's result, not the data's, and the control arm is what the curve is measured
    against. The warmup is a share of the run's steps rather than a count of epochs, because an
    epoch is four steps at the smallest budget and a hundred and sixty at the largest.

    Invariants: the epochs and the batch size are positive; the learning rate is positive and
    finite; the weight decay is finite and not negative; the warmup share lies in ``[0, 1)``;
    the final fraction lies in ``[0, 1]``.

    Attributes:
        epochs: Passes over the labelled windows.
        batch_size: Windows per optimiser step.
        learning_rate: Peak step size of the optimiser.
        weight_decay: Decoupled weight decay of the optimiser; zero for none.
        warmup_fraction: Share of the run's optimiser steps over which the rate climbs to its
            peak; zero for none.
        final_lr_fraction: Fraction of the peak the rate decays to by the last step; one for a
            constant rate.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    final_lr_fraction: float

    def __post_init__(self) -> None:
        for label, count in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if count < 1:
                raise InvalidAdaptationScheduleError(f"{label} must be positive, got {count}")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise InvalidAdaptationScheduleError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise InvalidAdaptationScheduleError(
                f"weight_decay must be finite and not negative, got {self.weight_decay}"
            )
        if not isfinite(self.warmup_fraction) or not 0.0 <= self.warmup_fraction < 1.0:
            raise InvalidAdaptationScheduleError(
                f"warmup_fraction must lie in [0, 1), got {self.warmup_fraction}"
            )
        if not isfinite(self.final_lr_fraction) or not 0.0 <= self.final_lr_fraction <= 1.0:
            raise InvalidAdaptationScheduleError(
                f"final_lr_fraction must lie in [0, 1], got {self.final_lr_fraction}"
            )

    def steps_per_epoch(self, windows: int) -> int:
        """Optimiser steps an epoch over ``windows`` labelled windows takes.

        Raises:
            InvalidAdaptationScheduleError: If there is no window to learn from.
        """
        if windows < 1:
            raise InvalidAdaptationScheduleError(f"an epoch must hold a window, got {windows}")
        return -(-windows // self.batch_size)

    def learning_rate_schedule(self, windows: int) -> LearningRateSchedule:
        """The rate over the whole run, in optimiser steps, for a sample of ``windows``.

        The warmup is rounded to whole steps and capped so that at least one step is left to
        decay over, which a run of a single step needs.

        Raises:
            InvalidAdaptationScheduleError: If there is no window to learn from.
        """
        total = self.epochs * self.steps_per_epoch(windows)
        return LearningRateSchedule(
            warmup_steps=min(round(self.warmup_fraction * total), total - 1),
            total_steps=total,
            final_fraction=self.final_lr_fraction,
        )
