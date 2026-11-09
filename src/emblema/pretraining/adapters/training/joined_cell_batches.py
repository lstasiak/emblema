from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from emblema.pretraining.adapters.objective.cell_tensors import CellTensors
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.joined_cell_windows import JoinedCellWindows
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.joined_cells import JoinedCells
from emblema.shared.adapters.arrays.joined_windows import JoinedWindow
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class JoinedCellBatches:
    """Cells of windows joined with those before them: masks over the full cells, then rejoined.

    A batch is joined from every window's cells first, which gives every cell of the joined
    window, empty ones included, for the masks to be drawn over and for a hidden cell's target,
    and the layout keeps a row for each channel the joined window observes, as a layout over a
    window's own channels does. Once the masks are drawn the hidden cells are emptied in the
    cells joined from and the windows joined again on the same rows, so a cell after a hidden one
    carries forward and counts from what the encoder can see, as laying the readings without the
    hidden ones would give.

    Attributes:
        joined: How the cells of a window's span are joined into its bins.
    """

    def __init__(self, bins: int) -> None:
        self.joined = JoinedCells(bins)
        self._layout = BinnedTokens(bins)

    def collate(self, windows: Sequence[JoinedWindow]) -> JoinedCellWindows:
        """``windows`` joined and laid out over the channels each observes.

        Raises:
            ValueError: If no window is given, or a window's cells do not cut into the bins.
        """
        if not windows:
            raise ValueError("a batch needs at least one window")
        cells = [self.joined.join(window.cells) for window in windows]
        kept = [(joined[..., BinnedTokens.PRESENCE] > 0.0).any(axis=-1) for joined in cells]
        laid = [
            JoinedCells.laid(joined[keep], window.rows[keep], own_channels=False)
            for joined, keep, window in zip(cells, kept, windows, strict=True)
        ]
        batch = TokenTensors.of(self._layout.batch(laid))
        return JoinedCellWindows(
            batch.features,
            batch.channel_ids,
            batch.timestamps,
            batch.timeless,
            batch.padding_mask,
            joined=tuple(windows),
            kept=tuple(kept),
        )

    def scored(self, batch: TokenTensors, masks: TokenMasks) -> CellTensors:
        """The full cells beside the cells joined again with the hidden ones emptied.

        Raises:
            ValueError: If ``batch`` was not collated here, so holds no windows to join again.
        """
        if not isinstance(batch, JoinedCellWindows):
            raise ValueError(
                "joined cells are scored from the cells they were joined from, which this batch "
                "has not"
            )
        hidden = masks.hidden.to("cpu")
        visible = self._layout.batch(
            [
                self._visible(window, keep, batch.hidden_of(row, hidden).numpy().astype(np.bool_))
                for row, (window, keep) in enumerate(zip(batch.joined, batch.kept, strict=True))
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

    def _visible(
        self, window: JoinedWindow, keep: NDArray[np.bool_], hidden: NDArray[np.bool_]
    ) -> LaidWindow:
        """``window`` joined again with its hidden cells emptied, on the rows the full one kept.

        A cell of the layout is one kept row in one bin, bin by bin with the rows within a bin;
        emptying it empties every cell of the windows the bin gathers for that row.
        """
        bins = self.joined.bins
        emptying = np.zeros((len(window.rows), bins), dtype=np.bool_)
        emptying[keep] = hidden.reshape(bins, int(keep.sum())).T
        emptied = window.cells.copy()
        each = window.cells.shape[1] // bins
        emptied[np.repeat(emptying, each, axis=1), : BinnedTokens.LOG_COUNT + 1] = 0.0
        joined = self.joined.join(emptied)
        return JoinedCells.laid(joined[keep], window.rows[keep], own_channels=False)
