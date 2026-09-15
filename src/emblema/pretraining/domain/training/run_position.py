from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidRunPositionError


@dataclass(frozen=True, kw_only=True)
class RunPosition:
    """Where a run stands: which epoch it is in, how much of it is consumed, how far it has come.

    The epoch is carried rather than derived, because it is what fixes the order the windows
    arrive in; the micro-batches consumed are what a resumed run has to skip before its first
    gradient, so that an interrupted run and an uninterrupted one see the same windows in the same
    order. Steps count optimiser steps, which is what the learning rate and the checkpoint policy
    are stated in.

    Invariants: nothing is negative.

    Attributes:
        epoch: Epoch being trained, counted from zero.
        batches: Micro-batches of that epoch already consumed.
        steps: Optimiser steps completed since the run began.
    """

    epoch: int
    batches: int
    steps: int

    def __post_init__(self) -> None:
        for label, count in (
            ("epoch", self.epoch),
            ("batches", self.batches),
            ("steps", self.steps),
        ):
            if count < 0:
                raise InvalidRunPositionError(f"{label} must not be negative, got {count}")

    @classmethod
    def start(cls) -> Self:
        """Where a run that has done nothing stands."""
        return cls(epoch=0, batches=0, steps=0)

    def after_batch(self, *, stepped: bool) -> Self:
        """One micro-batch further on, one optimiser step further where the gradient was applied."""
        return type(self)(
            epoch=self.epoch, batches=self.batches + 1, steps=self.steps + (1 if stepped else 0)
        )

    def next_epoch(self) -> Self:
        """The start of the following epoch, with the steps taken so far kept."""
        return type(self)(epoch=self.epoch + 1, batches=0, steps=self.steps)
