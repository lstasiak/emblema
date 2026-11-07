from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.training.joined_windows import JoinedWindow
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class JoinedCellWindows(TokenTensors):
    """A batch of joined windows laid out as cells, still holding the cells they were joined from.

    The five tensors are the full layout, every cell of every window, which is what the masks of
    a run are drawn over. The windows' cells stay beside them because once the masks are drawn
    the hidden cells are emptied and the windows joined again, so the cells after a hidden one
    carry forward and count from what remains visible. Lives on the host, where the layout is
    computed.

    Attributes:
        joined: The windows these cells were joined from, in the batch's row order.
        kept: Per window, which of its rows the layout kept: those some cell of it observes.
    """

    joined: Sequence[JoinedWindow]
    kept: Sequence[NDArray[np.bool_]]

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """The same cells on ``device``; the windows stay as they are, being no tensors."""
        moved = TokenTensors(
            self.features, self.channel_ids, self.timestamps, self.timeless, self.padding_mask
        ).to(device, dtype)
        return replace(
            self,
            features=moved.features,
            channel_ids=moved.channel_ids,
            timestamps=moved.timestamps,
            timeless=moved.timeless,
            padding_mask=moved.padding_mask,
        )

    def hidden_of(self, row: int, hidden: Tensor) -> Tensor:
        """``hidden`` cut to the cells of the window in ``row``, padding left out."""
        return hidden[row, : int((~self.padding_mask[row]).sum().item())]
