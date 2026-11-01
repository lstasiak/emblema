from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes


class FeatureDetector(Protocol):
    """What a runtime asks of a detector fitted over features, whatever it scores by.

    A seam between the runtimes that fit a detector and the detectors themselves, typed in the
    arrays they exchange: one detector a distance, or a head taught events, answers it.
    """

    def scores(
        self, rows: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """The score above the alarm of every row, in the order given."""

    def kept(self) -> RepresentationBytes:
        """The detector as it is kept, under the format its class reads back."""
