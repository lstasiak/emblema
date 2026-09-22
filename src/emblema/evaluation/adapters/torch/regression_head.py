import torch
from torch import Tensor, nn


class RegressionHead(nn.Module):
    """One number per window out of its pooled state: the head of a remaining-life task.

    Linear, because the question a probe asks is what the representation carries as it stands,
    and every other mode answers through the same head so that the modes differ in the backbone
    alone. The number is in units of the task's label ceiling, which whoever scores it undoes;
    a target of a few hundred cycles would otherwise dominate the first steps of every run. The
    bias starts where a predictor that knows nothing should, at the mean of the labels the run
    holds: the default draw puts the first answers anywhere in the range and a run of a few
    hundred steps spends a share of them walking the bias back.
    """

    def __init__(self, width: int, *, starting_at: float) -> None:
        """A head over states of ``width``, answering ``starting_at`` before any step.

        Args:
            width: Size of the pooled state.
            starting_at: The bias before training, in units of the label ceiling.
        """
        super().__init__()
        self.linear = nn.Linear(width, 1)
        with torch.no_grad():
            self.linear.bias.fill_(starting_at)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled).squeeze(-1)
