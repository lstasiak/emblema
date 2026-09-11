from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import DTypeLike, NDArray

from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID, Token, TokenWindow

# The continuous features of a token, in the order they occupy the last axis of ``features``:
# the normalised value and the gap since the previous token of the same channel.
N_FEATURES = 2


@dataclass(frozen=True)
class TokenBatch:
    """Windows laid out as the five arrays the encoder declares, padded to a common length.

    Windows differ in token count, so shorter ones are followed by padding: positions marked in
    ``padding_mask`` (``True`` = padding, the PyTorch convention, so the mask reaches attention
    unchanged) that carry ``PADDING_CHANNEL_ID`` and zeros everywhere else. Tokens keep their
    window's canonical order. ``windows`` reverses the layout, so a batch round-trips to the
    windows it was built from up to the precision of ``dtype``.

    Attributes:
        features: ``[batch, tokens, 2]`` — value and gap per token, floating.
        channel_ids: ``[batch, tokens]`` — vocabulary entry per token, int64; ``PADDING_CHANNEL_ID``
            at padding positions.
        timestamps: ``[batch, tokens]`` — position within the window per token, floating; zero at
            timeless and padding positions.
        timeless: ``[batch, tokens]`` — whether the token is a static feature, bool.
        padding_mask: ``[batch, tokens]`` — ``True`` where the position is padding, bool.
    """

    features: NDArray[np.floating]
    channel_ids: NDArray[np.int64]
    timestamps: NDArray[np.floating]
    timeless: NDArray[np.bool_]
    padding_mask: NDArray[np.bool_]

    @classmethod
    def from_windows(cls, windows: Sequence[TokenWindow], *, dtype: DTypeLike = np.float32) -> Self:
        """Lay ``windows`` out row by row, padding each to the longest.

        Args:
            windows: At least one window.
            dtype: Floating type of ``features`` and ``timestamps``; the encoder's input precision
                is a property of the model artifact, so the caller states it.

        Raises:
            ValueError: If no window is given.
        """
        if not windows:
            raise ValueError("a batch needs at least one window")
        rows, width = len(windows), max(len(window) for window in windows)
        features = np.zeros((rows, width, N_FEATURES), dtype=dtype)
        channel_ids = np.full((rows, width), PADDING_CHANNEL_ID, dtype=np.int64)
        timestamps = np.zeros((rows, width), dtype=dtype)
        timeless = np.zeros((rows, width), dtype=np.bool_)
        padding_mask = np.ones((rows, width), dtype=np.bool_)
        for row, window in enumerate(windows):
            count = len(window)
            features[row, :count, 0] = window.values
            features[row, :count, 1] = window.gaps
            channel_ids[row, :count] = window.channel_ids
            timestamps[row, :count] = window.times
            timeless[row, :count] = window.timeless
            padding_mask[row, :count] = False
        return cls(features, channel_ids, timestamps, timeless, padding_mask)

    @property
    def batch_size(self) -> int:
        return int(self.padding_mask.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.padding_mask.shape[1])

    def windows(self) -> tuple[TokenWindow, ...]:
        """The windows this batch lays out, padding removed, in canonical order.

        Raises:
            InvalidTokenWindowError: If a row does not hold a valid window, such as one without a
                single observed token.
        """
        return tuple(self._window(row) for row in range(self.batch_size))

    def _window(self, row: int) -> TokenWindow:
        observed = np.flatnonzero(~self.padding_mask[row])
        return TokenWindow.of(
            Token(
                channel_id=int(self.channel_ids[row, index]),
                value=float(self.features[row, index, 0]),
                time=float(self.timestamps[row, index]),
                gap=float(self.features[row, index, 1]),
                timeless=bool(self.timeless[row, index]),
            )
            for index in observed
        )
