from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


class SummaryBranch(nn.Module):
    """The window's statistics per channel, the trees' own table, as one state of a given width.

    Prepared from the windows the run draws alone, so nothing of the windows it is scored on
    shapes it. A column is kept where the drawn windows define at least two distinct values. Each
    is bounded to the range the drawn windows span and standardised by the mean and spread of
    the values they define. Imputing first would shrink the spread of a rarely measured channel
    to almost nothing, and send any scored window that holds it thousands of deviations out.
    Beyond the drawn range a tree is constant too, so the bound takes nothing a tree could use.
    A statistic a window cannot support lands at zero, the mean; whether a channel was measured
    at all is its count, which is always defined. Then one hidden layer, so the head reads them
    through a network rather than a linear map: a linear map over them is the logistic head
    already measured.

    The columns kept, their bounds and their moments are buffers, so a kept candidate holds them
    with its weights.
    """

    columns: Tensor
    lower: Tensor
    upper: Tensor
    centre: Tensor
    scale: Tensor

    def __init__(
        self,
        columns: Tensor,
        bounds: tuple[Tensor, Tensor],
        moments: tuple[Tensor, Tensor],
        *,
        width: int,
    ) -> None:
        super().__init__()
        self.register_buffer("columns", columns)
        self.register_buffer("lower", bounds[0])
        self.register_buffer("upper", bounds[1])
        self.register_buffer("centre", moments[0])
        self.register_buffer("scale", moments[1])
        self.hidden = nn.Linear(len(columns), width)
        self.activation = nn.GELU()

    @classmethod
    def fitted(cls, rows: NDArray[np.float64], *, width: int) -> Self:
        """A branch prepared over ``rows``, the statistics of the windows the run draws.

        Its layer is drawn from torch's current generator, so it is drawn wherever this is called.
        """
        kept = np.flatnonzero([np.unique(column[~np.isnan(column)]).size >= 2 for column in rows.T])
        chosen = rows[:, kept]

        def narrowed(values: NDArray[np.float64]) -> Tensor:
            return torch.from_numpy(values).to(torch.float32)

        return cls(
            torch.from_numpy(kept).to(torch.int64),
            (narrowed(np.nanmin(chosen, axis=0)), narrowed(np.nanmax(chosen, axis=0))),
            (narrowed(np.nanmean(chosen, axis=0)), narrowed(np.nanstd(chosen, axis=0))),
            width=width,
        )

    def forward(self, rows: Tensor) -> Tensor:
        bounded = torch.clamp(rows[:, self.columns], self.lower, self.upper)
        standard = torch.nan_to_num((bounded - self.centre) / self.scale, nan=0.0)
        hidden: Tensor = self.activation(self.hidden(standard))
        return hidden
