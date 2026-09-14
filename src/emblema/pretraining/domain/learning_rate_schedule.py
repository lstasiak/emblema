import math
from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidLearningRateScheduleError


@dataclass(frozen=True, kw_only=True)
class LearningRateSchedule:
    """How the learning rate moves over a run: a linear warmup, then a cosine decay to a floor.

    A factor of the peak rate per optimiser step, so the peak stays the optimiser's setting and the
    shape the run's. The decay keeps a verdict from being read off a model still circling its
    minimum; the warmup keeps a fresh model's first steps from undoing themselves. The warmup
    reaches the peak on its last step, so no step runs at zero; the decay starts at the peak and
    reaches the floor on the run's last step, or takes its one step at the peak where the warmup
    left it only one. Steps past the run stay at the floor. A pure function of the step: whoever
    trains holds the state.

    Invariants: at least one step in all; warmup steps are not negative and leave at least one step
    to decay, since a run that only warms up would never use the floor it states; the floor lies in
    ``[0, 1]``. A floor of one with no warmup is the constant rate.

    Attributes:
        warmup_steps: Steps over which the factor climbs to one.
        total_steps: Steps of the whole run.
        final_fraction: Factor the decay ends at, as a fraction of the peak.
    """

    warmup_steps: int
    total_steps: int
    final_fraction: float

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise InvalidLearningRateScheduleError(
                f"total_steps must be positive, got {self.total_steps}"
            )
        if not 0 <= self.warmup_steps < self.total_steps:
            raise InvalidLearningRateScheduleError(
                f"warmup_steps must lie in [0, total_steps) so that a step is left to decay, got "
                f"{self.warmup_steps} of {self.total_steps}"
            )
        if not 0.0 <= self.final_fraction <= 1.0:
            raise InvalidLearningRateScheduleError(
                f"final_fraction must lie in [0, 1], got {self.final_fraction}"
            )

    def factor(self, step: int) -> float:
        """The factor of the peak rate the optimiser uses at ``step``, counted from zero.

        Raises:
            InvalidLearningRateScheduleError: If ``step`` is negative.
        """
        if step < 0:
            raise InvalidLearningRateScheduleError(f"step must not be negative, got {step}")
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        decaying = self.total_steps - self.warmup_steps
        progress = min((step - self.warmup_steps) / max(decaying - 1, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_fraction + (1.0 - self.final_fraction) * cosine
