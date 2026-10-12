from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ObjectiveScore:
    """What one batch cost an objective, before it is divided: three sums over the same tokens.

    The error carries the graph the step is taken on; the trivial predictor's loss is what the
    same tokens would have cost a model that learnt nothing, read the same way, and is detached;
    the count is what divides either into a mean. Summed rather than averaged so that whoever
    accumulates several batches into one gradient adds the sums and divides once, and a batch
    that hid three tokens weighs a hundredth of one that hid three hundred.

    Attributes:
        error: The run's loss summed over the hidden tokens that were observed.
        trivial: The trivial predictor's loss summed over the same tokens.
        tokens: How many tokens were scored.
    """

    error: Tensor
    trivial: Tensor
    tokens: Tensor
