from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
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

    Fitted on windows with nothing hidden, where every token is a target and every other channel
    a visible feature, so that what it knows comes from the same data the model trains on.

    Attributes:
        coefficients: ``[entries, entries + 1]`` — a row of weights per target channel, the bias
            last. A channel the fit never saw has a row of zeros and predicts the channel mean.
    """

    coefficients: NDArray[np.float64]

    @classmethod
    def fitted(
        cls, batches: Iterable[TokenTensors], *, vocabulary_size: int, penalty: float = 1.0
    ) -> Self:
        """Fit one regression per channel over every observed token of ``batches``.

        Args:
            batches: Training windows, nothing hidden.
            vocabulary_size: Entries of the channel vocabulary, so that a row exists for every
                channel whether or not the batches show it.
            penalty: Weight of the ridge term, on every coefficient but the bias.

        Raises:
            ValueError: If ``vocabulary_size`` is not positive or ``penalty`` is negative.
        """
        if vocabulary_size < 1:
            raise ValueError(f"vocabulary size must be positive, got {vocabulary_size}")
        if penalty < 0:
            raise ValueError(f"penalty must not be negative, got {penalty}")
        entries = vocabulary_size + 1
        gram = np.zeros((entries, entries + 1, entries + 1))
        moment = np.zeros((entries, entries + 1))
        for batch in batches:
            arrays = WindowArrays.of(batch)
            for row in range(arrays.rows):
                observed = arrays.observed[row]
                series = arrays.series(row, observed)
                design = _design(series, arrays.times[row], entries)
                for channel in series:
                    targets = observed & (arrays.channel_ids[row] == channel)
                    rows = design[targets].copy()
                    rows[:, channel] = 0.0
                    gram[channel] += rows.T @ rows
                    moment[channel] += rows.T @ arrays.values[row, targets]
        ridge = penalty * np.eye(entries + 1)
        ridge[-1, -1] = 0.0
        coefficients = np.zeros((entries, entries + 1))
        for channel in range(entries):
            if moment[channel].any():
                coefficients[channel] = np.linalg.solve(gram[channel] + ridge, moment[channel])
        return cls(coefficients)

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
            series = arrays.series(row, ~hidden[row])
            design = _design(series, arrays.times[row], entries)
            targets = np.flatnonzero(hidden[row])
            channels = arrays.channel_ids[row, targets]
            rows = design[targets]
            rows[np.arange(len(targets)), channels] = 0.0
            prediction[row, targets] = np.einsum("tf,tf->t", rows, self.coefficients[channels])
        return torch.from_numpy(prediction).to(batch.features.device, batch.features.dtype)


def _design(
    series: Mapping[int, ChannelSeries], times: NDArray[np.float64], entries: int
) -> NDArray[np.float64]:
    """One row per token of the window: every channel's nearest visible value, then a bias."""
    design = np.zeros((len(times), entries + 1))
    for channel, line in series.items():
        design[:, channel] = line.nearest(times)
    design[:, -1] = 1.0
    return design
