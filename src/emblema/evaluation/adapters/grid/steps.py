"""Which step of an equal grid over a window an instant falls in, stated once for every grid.

A rule shared by the grid that lays a window out as an array and the one that hands it back as
tokens, so a reading lands in the same step whichever form a method reads.
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
