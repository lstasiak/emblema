import math
from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidLearningRateScheduleError


@dataclass(frozen=True, kw_only=True)
class LearningRateSchedule:
    """How the learning rate moves over a run: a linear warmup, then a cosine decay to a floor.

    Stated as a factor of the peak rate per optimiser step rather than as a rate, so that the peak
    stays a setting of the optimiser and the shape stays a setting of the run. A rate that never
    decays leaves the model circling its minimum at the size of its last steps, and a verdict read
    off such a run is read off whichever point of the circle the last epoch landed on; a rate that
    starts at its peak lets the first steps of a freshly initialised model undo themselves. The
    shape here removes both without adding a knob a run could be tuned on: warmup length, total
    length and the floor the decay ends at.

    The warmup climbs to the peak by the last of its steps, so no step is taken at a rate of zero.
    The decay starts at the peak on the first step after the warmup and reaches the floor on the
    last step of the run; steps past the run stay at the floor. Held with its state by whoever
    trains — the schedule itself is a pure function of the step.

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
