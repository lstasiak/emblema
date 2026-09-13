import numpy as np
import torch
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class LinearInterpolationBaseline:
    """What a hidden token's own channel says about it: a line between its visible neighbours.

    The trivial answer for a token hidden inside a block or on its own. A model that does no
    better than this on such tokens has learnt to smooth a channel, not to read the window. A
    channel with no visible token left offers no line to draw, and a timeless token has no
    neighbours in time; both predict the channel mean, zero after normalisation, and are the
    business of the cross-channel baseline.
    """

    def predict(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The baseline's value at every hidden token, zero elsewhere."""
        arrays = WindowArrays.of(batch)
        hidden = masks.hidden.cpu().numpy() & arrays.observed
        timeless = batch.timeless.cpu().numpy()
        prediction = np.zeros_like(arrays.values)
        for row in range(arrays.rows):
            series = arrays.series(row, ~hidden[row])
            for channel, line in series.items():
                targets = hidden[row] & ~timeless[row] & (arrays.channel_ids[row] == channel)
                if targets.any():
                    prediction[row, targets] = line.interpolate(arrays.times[row, targets])
        return torch.from_numpy(prediction).to(batch.features.device, batch.features.dtype)
