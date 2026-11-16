"""Which step of an equal division of a window an instant falls in, stated once for every layout.

A rule shared by the grids that lay a window out as an array or hand it back as tokens and by
the cells of channel and time bin, so a reading lands in the same step whichever form a method
reads, and a corpus sampled on the very edges of the steps is not laid out by its rounding.
"""

import numpy as np
from numpy.typing import NDArray

# A token's time is stored in single precision, so a reading at the start of a step can land a
# hair before it — k / 50 is 0.0199999995 — and a floor would put it one step early, leaving its
# own step empty and doubling the one before. A tolerance far above the rounding of single
# precision and far below any spacing of readings puts every reading back.
TOLERANCE = 1e-3


def steps_of(times: NDArray[np.float64], steps: int) -> NDArray[np.int64]:
    """The step each of ``times``, fractions of the window, falls in; the end in the last."""
    return np.minimum(np.floor(times * steps + TOLERANCE).astype(np.int64), steps - 1)
