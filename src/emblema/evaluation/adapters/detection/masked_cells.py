from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaskedCells:
    """One draw of a backbone's masks over one window's cells, and what its decoder answered.

    Every array stands cell for cell in the order the layout gives a window's cells: static
    cells first, then bin by bin with the channels within a bin.

    Attributes:
        full: ``[cells, features]`` — the cells laid out from every reading.
        visible: ``[cells, features]`` — the same cells laid out from the readings left visible.
        hidden: ``[cells]`` — the cells the draw hid.
        channel_ids: ``[cells]`` — the channel of every cell.
        timeless: ``[cells]`` — the cells of static features, which have no time.
        answers: ``[cells, 5]`` — the decoder's mean, minimum, maximum, first and last of every
            cell, in normalised units; ``None`` where the draw was asked for without them.
    """

    full: NDArray[np.float64]
    visible: NDArray[np.float64]
    hidden: NDArray[np.bool_]
    channel_ids: NDArray[np.int64]
    timeless: NDArray[np.bool_]
    answers: NDArray[np.float64] | None
