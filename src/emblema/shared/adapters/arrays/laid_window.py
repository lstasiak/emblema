from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike, NDArray

from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID


@dataclass(frozen=True)
class LaidWindow:
    """One window as the columns a layout gives it, laid once and batched as often as needed.

    A run reads the same windows on every step of every epoch, so a layout lays each out once
    and a batch is only the padding of the rows asked for. The columns are the token batch's,
    one window at a time, with the feature axis as wide as the layout makes it.

    Attributes:
        features: ``[tokens, features]``, floating.
        channel_ids: ``[tokens]``, int64.
        times: ``[tokens]``, floating.
        timeless: ``[tokens]``, bool.
    """

    features: NDArray[np.float64]
    channel_ids: NDArray[np.int64]
    times: NDArray[np.float64]
    timeless: NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.channel_ids)

    @classmethod
    def padded(cls, laid: Sequence["LaidWindow"], *, features: int, dtype: DTypeLike) -> TokenBatch:
        """``laid`` row by row, each padded to the longest, the feature axis ``features`` wide.

        Raises:
            ValueError: If no window is given.
        """
        if not laid:
            raise ValueError("a batch needs at least one window")
        rows, width = len(laid), max(len(window) for window in laid)
        columns = np.zeros((rows, width, features), dtype=dtype)
        channel_ids = np.full((rows, width), PADDING_CHANNEL_ID, dtype=np.int64)
        timestamps = np.zeros((rows, width), dtype=dtype)
        timeless = np.zeros((rows, width), dtype=np.bool_)
        padding_mask = np.ones((rows, width), dtype=np.bool_)
        for row, window in enumerate(laid):
            count = len(window)
            columns[row, :count] = window.features
            channel_ids[row, :count] = window.channel_ids
            timestamps[row, :count] = window.times
            timeless[row, :count] = window.timeless
            padding_mask[row, :count] = False
        return TokenBatch(columns, channel_ids, timestamps, timeless, padding_mask)
