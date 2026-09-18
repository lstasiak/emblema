from torch import Tensor, nn


class RegressionHead(nn.Module):
    """One number per window out of its pooled state: the head of a remaining-life task.

    Linear, because the question a probe asks is what the representation carries as it stands,
    and every other mode answers through the same head so that the modes differ in the backbone
    alone. The number is in units of the task's label ceiling, which whoever scores it undoes;
    a target of a few hundred cycles would otherwise dominate the first steps of every run.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled).squeeze(-1)
