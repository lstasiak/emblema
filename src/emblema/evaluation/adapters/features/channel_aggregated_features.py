from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.features.window_spectrum import WindowSpectrum
from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from emblema.shared.kernel.tokens import TokenWindow


class ChannelAggregatedFeatures:
    """Reads a window as its per-channel statistics and spectra, summarised across its channels.

    The only reading in this system under which a classical method transfers at all. Summarising
    across the channels throws away which channel said what and keeps how the channels behaved,
    and what is left has the same width whether the corpus has eight channels or forty-four. A
    candidate fitted this way over several corpora and scored on another is the classical
    alternative explanation for a pretrained encoder's transfer, which is why the project has to
    measure it rather than assert it cannot exist.

    The spectrum is summarised the same way, because a classical method that transfers must be
    offered what an engineer would compare two machines by, and the share of power in an octave
    is as meaningful for a turbofan as for a patient. Its frequencies are counted per window,
    so two corpora of different window lengths compare the same count of cycles, not the same
    physical period — which is the only comparison a layout-free reading can make.

    Three columns stand outside the summary and describe the window itself: how much was
    observed, over how many channels, and how much of it was static. They are what tells a
    sparsely observed stay from a densely sampled engine when every summarised statistic is
    otherwise alike.
    """

    ACROSS: ClassVar[tuple[str, ...]] = ("mean", "deviation", "minimum", "maximum")
    OF_WINDOW: ClassVar[tuple[str, ...]] = ("tokens", "channels", "timeless")

    @property
    def width(self) -> int:
        summarised = len(WindowStatistics.NAMES) + len(WindowSpectrum.NAMES)
        return summarised * len(self.ACROSS) + len(self.OF_WINDOW)

    def names(self) -> tuple[str, ...]:
        """What each column holds, so a stored candidate can be read a year later."""
        return (
            tuple(f"{across}_{name}" for name in WindowStatistics.NAMES for across in self.ACROSS)
            + tuple(
                f"{across}_spectrum_{name}"
                for name in WindowSpectrum.NAMES
                for across in self.ACROSS
            )
            + tuple(f"window_{name}" for name in self.OF_WINDOW)
        )

    def of(self, windows: Sequence[TokenWindow]) -> NDArray[np.float64]:
        """The windows as one row each, in the order given."""
        rows = np.empty((len(windows), self.width), dtype=np.float64)
        for index, window in enumerate(windows):
            statistics = WindowStatistics.of(window)
            rows[index] = np.concatenate(
                (
                    self._across(statistics.rows),
                    self._across(WindowSpectrum.of(window).rows),
                    self._of_window(window, statistics),
                )
            )
        return rows

    @staticmethod
    def _across(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each column summarised over the channels that have it, ignoring those that do not.

        A column no channel of the window supports — a slope where every channel was observed
        once, a spectrum where none was observed thrice — stays absent rather than becoming a
        number, for the same reason it was absent per channel.
        """
        known = ~np.isnan(rows)
        counts = known.sum(axis=0)
        held = np.maximum(counts, 1)
        somewhere = counts > 0
        with np.errstate(invalid="ignore"):
            mean = np.where(somewhere, np.where(known, rows, 0.0).sum(axis=0) / held, np.nan)
            spread = np.where(known, (rows - mean) ** 2, 0.0).sum(axis=0) / held
            summarised = np.column_stack(
                (
                    mean,
                    np.where(somewhere, np.sqrt(spread), np.nan),
                    np.where(somewhere, np.where(known, rows, np.inf).min(axis=0), np.nan),
                    np.where(somewhere, np.where(known, rows, -np.inf).max(axis=0), np.nan),
                )
            )
        return np.asarray(summarised.ravel(), dtype=np.float64)

    @staticmethod
    def _of_window(window: TokenWindow, statistics: WindowStatistics) -> NDArray[np.float64]:
        return np.array(
            [len(window), len(statistics.channels), sum(window.timeless)], dtype=np.float64
        )
