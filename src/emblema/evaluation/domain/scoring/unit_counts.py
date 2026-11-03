from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class UnitCounts:
    """How many times each of several draws counts each unit, one row per draw.

    Built once for a batch of draws and read by every ranking of a comparison, so the units are
    counted once however many repeats the sides hold.

    Attributes:
        rows: The count of each unit (column) under each draw (row).
    """

    rows: NDArray[np.float64]

    @classmethod
    def of_weights(cls, weights: Sequence[int]) -> Self:
        """One draw that counts unit ``i`` ``weights[i]`` times."""
        return cls(rows=np.asarray([weights], dtype=np.float64))

    @classmethod
    def of_picks(cls, draws: Sequence[Sequence[int]], units: int) -> Self:
        """The draws, each a sequence of unit indices, a unit counted as often as it is picked."""
        return cls(
            rows=np.asarray(
                [np.bincount(np.asarray(p, dtype=np.intp), minlength=units) for p in draws],
                dtype=np.float64,
            )
        )
