from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

import torch
from torch import Tensor

from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class CellWindows(TokenTensors):
    """A batch of windows laid out as cells, still holding the windows it was laid from.

    The five tensors are the full layout, every cell of every window, which is what the masks
    of a run are drawn over. The windows stay beside them because once the masks are drawn the
    cells have to be laid out again from the readings the masks left visible, and that is done
    from the readings, not from the cells. Lives on the host, where the layout is computed: the
    batch a model is called with is made from it and moved, not this.

    Attributes:
        windows: The windows these cells were laid from, in the batch's row order.
    """

    windows: Sequence[TokenWindow]

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
