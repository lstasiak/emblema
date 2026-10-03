import math
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

from emblema.evaluation.domain.exceptions import DivergedAdaptationError
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.shared.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler

# What the model is asked per batch: the indices of the windows to answer, in sample order.
Forward = Callable[[Sequence[int]], Tensor]
# What a batch's answers are held to against what they were taught.
Loss = Callable[[Tensor, Tensor], Tensor]
# Told the index of the epoch just finished; answers whether the run stops here.
AfterEpoch = Callable[[int], bool]


class ScheduledTraining:
    """The loop every network of this context learns a task by, under one schedule and seed.

    One loop rather than one per network, because a compute budget shared between candidates is
    only shared if they spend it the same way: the same optimiser, the same shape of rate, the
    same seeded order of windows and the same refusal of a loss that stops being finite. What a
    candidate is made of reaches the loop as the parameters it may change and a function that
    answers a batch; what the task's target is reaches it as the loss the answers descend.
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
        *,
        loss: Loss,
        epochs: int | None = None,
        after_epoch: AfterEpoch | None = None,
    ) -> list[float]:
        """The schedule's epochs over the targets, in the seeded order; the mean loss of each.

        The epochs are the schedule's over this many targets: the stated ones, or more where the
        floor of steps asks for them; or ``epochs``, where a run learning from part of a sample
        is held to the whole sample's count. The rate follows the schedule's shape step by step
        over that many epochs. ``after_epoch``, if given, is called with each epoch's index once
        its loss is in and may end the run early by returning ``True``; the model is put back
        into training afterwards, whatever it was called in.

        Raises:
            DivergedAdaptationError: If a batch's loss stops being finite.
        """
        schedule = self._schedule
        planned = schedule.epochs_over(len(targets)) if epochs is None else epochs
        optimiser = torch.optim.AdamW(
            trainable, lr=schedule.learning_rate, weight_decay=schedule.weight_decay
        )
        rate = torch.optim.lr_scheduler.LambdaLR(
            optimiser, schedule.learning_rate_schedule(len(targets), planned).factor
        )
        order = SeededShuffleSampler(len(targets), seed=self._seed)
        losses = []
        model.train()
        for epoch in range(planned):
            order.set_epoch(epoch)
            positions = list(order)
            total = 0.0
            for start in range(0, len(positions), schedule.batch_size):
                indices = positions[start : start + schedule.batch_size]
                batch_loss = loss(forward(indices), targets[indices])
                mean = float(batch_loss.detach())
                if not math.isfinite(mean):
                    raise DivergedAdaptationError(f"the loss of a batch in epoch {epoch} is {mean}")
                optimiser.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimiser.step()
                rate.step()
                total += mean * len(indices)
            losses.append(total / len(positions))
            if after_epoch is not None:
                stop = after_epoch(epoch)
                model.train()
                if stop:
                    break
        return losses
