from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.latent_prediction_loss import LatentPredictionLoss
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class LatentChannelBaseline(ABC):
    """A trivial answer to the objective that predicts states, read along the token's channel.

    What the line through visible values is to the objective that predicts values: an answer
    made of the teacher's states of the channel's visible tokens alone, so a predictor that does
    no better has learnt to carry the teacher along a channel, not to read the window. Each
    subclass says how the visible neighbours' states make the answer. A channel with no visible
    token, and a timeless token, have no neighbours in time: both answer with the mean state of
    the window's visible tokens, which is what the trivial predictor answers everywhere.
    """

    def predict(self, targets: Tensor, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The baseline's state at every hidden token, the visible mean elsewhere."""
        arrays = WindowArrays.of(batch)
        states = targets.detach().to("cpu", torch.float64).numpy()
        hidden = masks.hidden.cpu().numpy() & arrays.observed
        timeless = batch.timeless.cpu().numpy()
        # The mean is one vector a window broadcast over its positions; written into token by
        # token below, it has to own a copy per position first.
        prediction = (
            LatentPredictionLoss.visible_mean(targets, batch, masks)
            .detach()
            .to("cpu", torch.float64)
            .contiguous()
            .numpy()
        )
        for row in range(arrays.rows):
            for channel, line in arrays.series(row, ~hidden[row]).items():
                at = hidden[row] & ~timeless[row] & (arrays.channel_ids[row] == channel)
                if at.any():
                    prediction[row, at] = self._along(line, arrays.times[row, at], states[row])
        return torch.from_numpy(prediction).to(targets.device, targets.dtype)

    @abstractmethod
    def _along(
        self, line: ChannelSeries, at: NDArray[np.float64], states: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The answer at ``at`` from the channel's visible tokens and the row's states."""
