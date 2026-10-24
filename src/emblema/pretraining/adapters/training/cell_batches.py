from collections.abc import Sequence

import numpy as np
import torch

from emblema.pretraining.adapters.objective.cell_tensors import CellTensors
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.cell_windows import CellWindows
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow


class CellBatches:
    """One token per channel and span: masks over the full cells, the encoder fed the visible ones.

    A batch is laid out from every reading first, which gives every cell of every window, empty
    ones included, for the masks to be drawn over and for a hidden cell's target. Once the masks
    are drawn the same windows are laid out again without the readings of the hidden cells, on
    the same rows and bins, and the two layouts travel together: an empty cell beside a hidden
    one then carries forward and counts from what the encoder can see, and never from what was
    hidden.

    Attributes:
        layout: The cells a window is laid out as; the window's own channels, as a pretrained
            encoder reads every corpus.
    """

    def __init__(self, layout: BinnedTokens) -> None:
        self.layout = layout

    def collate(self, windows: Sequence[TokenWindow]) -> CellWindows:
        cells = TokenTensors.of(self.layout.batch(self.layout.lay(windows)))
        return CellWindows(
            cells.features,
            cells.channel_ids,
            cells.timestamps,
            cells.timeless,
            cells.padding_mask,
            windows=tuple(windows),
        )

    def scored(self, batch: TokenTensors, masks: TokenMasks) -> CellTensors:
        """The full cells beside the cells laid out without the hidden ones' readings.

        Raises:
            ValueError: If ``batch`` was not collated here, so holds no windows to lay out.
        """
        if not isinstance(batch, CellWindows):
            raise ValueError(
                "cells are scored from the windows they were laid from, which this batch has not"
            )
        hidden = masks.hidden.to("cpu")
        visible = self.layout.batch(
            [
                self.layout.without(window, batch.hidden_of(row, hidden).numpy().astype(np.bool_))
                for row, window in enumerate(batch.windows)
            ],
            dtype=batch.features.numpy().dtype,
        )
        return CellTensors(
            batch.features,
            batch.channel_ids,
            batch.timestamps,
            batch.timeless,
            batch.padding_mask,
            visible=torch.from_numpy(visible.features),
        )
