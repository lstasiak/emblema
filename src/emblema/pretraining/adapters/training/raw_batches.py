from collections.abc import Sequence

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow


class RawBatches:
    """One token per reading: the batch the masks are drawn over is the batch that is scored.

    What every run did before a token could be a cell, named as a layout so the runtime asks one
    object either way; the objective hides the hidden readings from the encoder itself.
    """

    def collate(self, windows: Sequence[TokenWindow]) -> TokenTensors:
        return TokenTensors.from_windows(windows)

    def scored(self, batch: TokenTensors, masks: TokenMasks) -> TokenTensors:
        return batch
