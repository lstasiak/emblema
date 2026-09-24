from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.features.window_spectrum import WindowSpectrum
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.kernel.tokens import TokenWindow


class SpectralFeatures:
    """Reads a window as one block of spectral summaries per channel of the corpus.

    The baseline an engineer with a background in instrumentation expects first: how much of a
    channel's variance sits in each octave, where its power is centred, how evenly it is spread
    and how strongly one frequency stands out. Laid out channel by channel, so it is as bound to
    the corpus as the statistics are, and a channel with no spectrum in a window — too few
    readings, a static feature, or no reading at all — leaves its block absent.
    """

    def __init__(self, channels: int) -> None:
        """Read windows of a corpus whose vocabulary runs from one to ``channels``.

        Raises:
            ValueError: If the corpus is said to have no channel.
        """
        if channels < 1:
            raise ValueError(f"a corpus has at least one channel, got {channels}")
        self._channels = channels

    @property
    def width(self) -> int:
        return self._channels * len(WindowSpectrum.NAMES)

    def names(self) -> tuple[str, ...]:
        """What each column holds, so a stored candidate can be read a year later."""
        return tuple(
            f"channel_{channel}_spectrum_{name}"
            for channel in range(1, self._channels + 1)
            for name in WindowSpectrum.NAMES
        )

    def of(self, windows: Sequence[TokenWindow]) -> NDArray[np.float64]:
        """The windows as one row each, in the order given.

        Raises:
            UnreadableTaskCorpusError: If a window holds a channel the corpus does not name,
                which means the windows and the manifest are not of the same publication.
        """
        rows = np.empty((len(windows), self.width), dtype=np.float64)
        for index, window in enumerate(windows):
            rows[index] = self._row(WindowSpectrum.of(window))
        return rows

    def _row(self, spectrum: WindowSpectrum) -> NDArray[np.float64]:
        block = np.full((self._channels, len(WindowSpectrum.NAMES)), np.nan)
        if len(spectrum.channels):
            highest = int(spectrum.channels[-1])
            if highest > self._channels:
                raise UnreadableTaskCorpusError(
                    f"a window holds channel {highest} and the corpus names {self._channels}"
                )
            block[spectrum.channels - 1] = spectrum.rows
        return block.ravel()
