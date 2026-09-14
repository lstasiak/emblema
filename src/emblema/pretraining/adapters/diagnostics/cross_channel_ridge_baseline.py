from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.ridge_regression import (
    nearest_values,
    solved,
    validate,
)
from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class CrossChannelRidgeBaseline:
    """What the other channels say about a hidden token: one ridge regression per channel.

    The regressors are the values the other channels show at their instants nearest to the token.

    The trivial answer for a channel hidden whole, where there is nothing of the channel itself to
    interpolate. A model that does no better than this on such tokens has learnt a linear
    cross-talk between sensors, which any regression would have found. The design row of a token
    holds one column per entry of the vocabulary — the nearest visible value of that channel in
    the window, zero where the channel shows nothing — and a bias; the token's own column is zero,
    so a channel never explains itself. One penalty for every coefficient but the bias: a baseline
    is not tuned.

    Fitted under the masks the strategy draws, on training windows, so that its regressors go
    missing as often and as far as they will when it predicts: a regression fitted on windows with
    nothing hidden learns weights for neighbours that are always there, then meets windows where a
    whole channel of them is zero and the nearest visible value of another lies half a window away,
    and is weaker than the trivial answer it stands for. Every observed token is a target, hidden
    or not — its own column is zeroed either way, and the other channels are masked independently
    of it — so that what it knows comes from the same data the model trains on.

    Attributes:
        coefficients: ``[entries, entries + 1]`` — a row of weights per target channel, the bias
            last. A channel the fit never saw has a row of zeros and predicts the channel mean.
    """

    coefficients: NDArray[np.float64]

    @classmethod
    def fitted(
        cls,
        batches: Iterable[tuple[TokenTensors, TokenMasks]],
        *,
        vocabulary_size: int,
        penalty: float = 1.0,
    ) -> Self:
        """Fit one regression per channel over every observed token of ``batches``.

        Args:
            batches: Training windows, each batch with the masks drawn over it; the regressors are
                read from the tokens the masks leave visible.
            vocabulary_size: Entries of the channel vocabulary, so that a row exists for every
                channel whether or not the batches show it.
            penalty: Weight of the ridge term, on every coefficient but the bias.

        Raises:
            ValueError: If ``vocabulary_size`` is not positive or ``penalty`` is negative.
        """
        validate(vocabulary_size, penalty)
        entries = vocabulary_size + 1
        gram = np.zeros((entries, entries + 1, entries + 1))
        moment = np.zeros((entries, entries + 1))
        for batch, masks in batches:
            arrays = WindowArrays.of(batch)
            visible = arrays.observed & ~masks.hidden.cpu().numpy()
            for row in range(arrays.rows):
                observed = arrays.observed[row]
                design = nearest_values(
                    arrays.series(row, visible[row]), arrays.times[row], entries
                )
                for channel in np.unique(arrays.channel_ids[row, observed]):
                    targets = observed & (arrays.channel_ids[row] == channel)
                    rows = design[targets].copy()
                    rows[:, channel] = 0.0
                    gram[channel] += rows.T @ rows
                    moment[channel] += rows.T @ arrays.values[row, targets]
        return cls(solved(gram, moment, penalty))

    def predict(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The baseline's value at every hidden token, zero elsewhere.

        Read from the visible tokens of the other channels: the hidden ones are not there to
        be read.
        """
        arrays = WindowArrays.of(batch)
        hidden = masks.hidden.cpu().numpy() & arrays.observed
        entries = self.coefficients.shape[0]
        prediction = np.zeros_like(arrays.values)
        for row in range(arrays.rows):
            if not hidden[row].any():
                continue
            design = nearest_values(arrays.series(row, ~hidden[row]), arrays.times[row], entries)
            targets = np.flatnonzero(hidden[row])
            channels = arrays.channel_ids[row, targets]
            rows = design[targets]
            rows[np.arange(len(targets)), channels] = 0.0
            prediction[row, targets] = np.einsum("tf,tf->t", rows, self.coefficients[channels])
        return torch.from_numpy(prediction).to(batch.features.device, batch.features.dtype)
