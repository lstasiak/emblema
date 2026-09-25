import math
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from emblema.evaluation.domain.exceptions import DivergedAdaptationError
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.shared.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler

# What the model is asked per batch: the indices of the windows to answer, in sample order.
Forward = Callable[[Sequence[int]], Tensor]


class ScheduledTraining:
    """The loop every network of this context learns a task by, under one schedule and seed.

    One loop rather than one per network, because a compute budget shared between candidates is
    only shared if they spend it the same way: the same optimiser, the same shape of rate, the
    same seeded order of windows and the same refusal of a loss that stops being finite. What a
    candidate is made of reaches the loop as the parameters it may change and a function that
    answers a batch.
    """

    def __init__(self, schedule: AdaptationSchedule, seed: int) -> None:
        self._schedule = schedule
        self._seed = seed

    def losses(
        self,
        model: nn.Module,
        trainable: Iterable[nn.Parameter],
        forward: Forward,
        targets: Tensor,
    ) -> list[float]:
        """The schedule's epochs over the targets, in the seeded order; the mean loss of each.

        The epochs are the schedule's over this many targets: the stated ones, or more where the
        floor of steps asks for them. The rate follows the schedule's shape step by step.

        Raises:
            DivergedAdaptationError: If a batch's loss stops being finite.
        """
        schedule = self._schedule
        optimiser = torch.optim.AdamW(
            trainable, lr=schedule.learning_rate, weight_decay=schedule.weight_decay
        )
        rate = torch.optim.lr_scheduler.LambdaLR(
            optimiser, schedule.learning_rate_schedule(len(targets)).factor
        )
        order = SeededShuffleSampler(len(targets), seed=self._seed)
        losses = []
        model.train()
        for epoch in range(schedule.epochs_over(len(targets))):
            order.set_epoch(epoch)
            positions = list(order)
            total = 0.0
            for start in range(0, len(positions), schedule.batch_size):
                indices = positions[start : start + schedule.batch_size]
                loss = mse_loss(forward(indices), targets[indices])
                mean = float(loss.detach())
                if not math.isfinite(mean):
                    raise DivergedAdaptationError(f"the loss of a batch in epoch {epoch} is {mean}")
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                rate.step()
                total += mean * len(indices)
            losses.append(total / len(positions))
        return losses
