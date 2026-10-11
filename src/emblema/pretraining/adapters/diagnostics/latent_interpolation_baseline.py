import numpy as np
from numpy.typing import NDArray

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
from emblema.pretraining.adapters.diagnostics.latent_channel_baseline import LatentChannelBaseline


class LatentInterpolationBaseline(LatentChannelBaseline):
    """A line between the states of the visible neighbours on either side of the hidden instant.

    The matched baseline for a block, a single token or the window's tail, where past the last
    visible token the last state is carried forward: the same line the value objective is read
    against, drawn between states instead of values.
    """

    def _along(
        self, line: ChannelSeries, at: NDArray[np.float64], states: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        before, after, weight = line.neighbours(at)
        blended: NDArray[np.float64] = (
            states[before] * (1.0 - weight)[:, None] + states[after] * weight[:, None]
        )
        return blended
