from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidComputeBudgetError
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


@dataclass(frozen=True, kw_only=True)
class ComputeBudget:
    """What one cell is allowed to spend learning, stated so two candidates can be held to it.

    Only what decides how much arithmetic a run does, and nothing about how it learns: two
    candidates may differ in their rate, their decay and how many weights they move, and still
    be spending the same budget. Rate and decay are method, and holding methods to a common
    value would compare something nobody asked about.

    Equality is the whole point: candidates share a budget when their budgets are equal, so the
    value carries counts rather than an estimate of time, which no two machines agree on.

    Invariants: at least one epoch and one window per batch; the floor of steps is not negative.

    Attributes:
        epochs: Passes over the labelled windows, at least.
        min_steps: Optimiser steps a run takes at least, whatever its sample holds.
        batch_size: Windows per optimiser step.
    """

    epochs: int
    min_steps: int
    batch_size: int

    def __post_init__(self) -> None:
        for label, count in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if count < 1:
                raise InvalidComputeBudgetError(f"{label} must be positive, got {count}")
        if self.min_steps < 0:
            raise InvalidComputeBudgetError(f"min_steps must not be negative, got {self.min_steps}")

    @classmethod
    def of(cls, schedule: AdaptationSchedule) -> Self:
        """The budget a network learning under ``schedule`` spends.

        Stated once, so that two networks under one schedule hold equal budgets by construction
        rather than because two copies of the same three fields happened to agree.
        """
        return cls(
            epochs=schedule.epochs, min_steps=schedule.min_steps, batch_size=schedule.batch_size
        )

    def __str__(self) -> str:
        return f"{self.epochs} epochs of {self.batch_size} windows, at least {self.min_steps} steps"
