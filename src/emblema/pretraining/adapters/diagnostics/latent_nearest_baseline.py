import numpy as np
from numpy.typing import NDArray

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
from emblema.pretraining.adapters.diagnostics.latent_channel_baseline import LatentChannelBaseline


class LatentNearestBaseline(LatentChannelBaseline):
    """The state of the visible token nearest in time within the hidden token's channel.

    Cruder than the line: no blending, one neighbour's state copied over. A predictor whose
    answer is this has read nothing of the window but which token is closest.
    """

    def _along(
        self, line: ChannelSeries, at: NDArray[np.float64], states: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        nearest: NDArray[np.float64] = states[line.nearest_positions(at)]
        return nearest
