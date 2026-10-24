from collections.abc import Sequence
from typing import Protocol

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow


class BatchLayout(Protocol):
    """How a run lays windows out: the batch its masks are drawn over, and the batch it scores.

    A seam of the training runtime, not a port: nothing in a use case sees it, and it speaks
    tensors. Under one token per reading the two are one batch: the masks are drawn over the
    readings and the objective hides the hidden ones itself. Under cells the masks are drawn over
    the full layout and the batch the objective scores carries a second layout beside it, from
    the readings the masks left visible, so the layout is computed where the readings are, on
    the host, before anything moves.
    """

    def collate(self, windows: Sequence[TokenWindow]) -> TokenTensors:
        """``windows`` as the batch a run draws its masks over.

        Raises:
            ValueError: If no window is given.
        """
        ...

    def scored(self, batch: TokenTensors, masks: TokenMasks) -> TokenTensors:
        """The batch the objective scores under ``masks``, on the host.

        Raises:
            ValueError: If ``batch`` is not one this layout collated.
        """
        ...
