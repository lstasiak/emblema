from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.tokens import N_FEATURES, TokenWindow


class RawTokens:
    """The layout every encoder was trained under: one token per reading, as the window holds it.

    The batch is what ``TokenBatch.from_windows`` lays out, value by value, named as a layout so
    a run that may lay windows out otherwise asks one object either way.
    """

    @property
    def features(self) -> int:
        return N_FEATURES

    def lay(self, windows: Sequence[TokenWindow]) -> list[LaidWindow]:
        return [
            LaidWindow(
                features=np.column_stack(
                    (
                        np.asarray(window.values, dtype=np.float64),
                        np.asarray(window.gaps, dtype=np.float64),
                    )
                ),
                channel_ids=np.asarray(window.channel_ids, dtype=np.int64),
                times=np.asarray(window.times, dtype=np.float64),
                timeless=np.asarray(window.timeless, dtype=np.bool_),
            )
            for window in windows
        ]

    def batch(self, laid: Sequence[LaidWindow], *, dtype: DTypeLike = np.float32) -> TokenBatch:
        return LaidWindow.padded(laid, features=N_FEATURES, dtype=dtype)
