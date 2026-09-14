"""What the linear baselines share: the regressors a window offers a token, and the ridge solve.

Two baselines regress a hidden token on what the rest of its window shows, and both would
otherwise build the same rows and solve the same normal equations in two places that could drift
apart. The rows are built from visible tokens only — a regressor read from a hidden token would be
the answer — and the solve leaves the bias unpenalised, because a baseline is not tuned and the
mean of a channel is not a weight to shrink.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries


def validate(vocabulary_size: int, penalty: float) -> None:
    """Refuse a fit over no vocabulary or with a penalty that rewards large weights.

    Raises:
        ValueError: If ``vocabulary_size`` is not positive or ``penalty`` is negative.
    """
    if vocabulary_size < 1:
        raise ValueError(f"vocabulary size must be positive, got {vocabulary_size}")
    if penalty < 0:
        raise ValueError(f"penalty must not be negative, got {penalty}")


def nearest_values(
    series: Mapping[int, ChannelSeries], times: NDArray[np.float64], entries: int
) -> NDArray[np.float64]:
    """One row per token of the window: every channel's nearest visible value, then a bias.

    A channel that shows nothing in the window contributes zero — its mean, once normalised.
    """
    design = np.zeros((len(times), entries + 1))
    for channel, line in series.items():
        design[:, channel] = line.nearest(times)
    design[:, -1] = 1.0
    return design


def solved(
    gram: NDArray[np.float64], moment: NDArray[np.float64], penalty: float
) -> NDArray[np.float64]:
    """Ridge weights for every system of stacked normal equations, the last coefficient a bias.

    ``gram`` is ``[systems, features, features]`` and ``moment`` ``[systems, features]``, the bias
    last in both. A system that saw no row — its bias column counts the rows it saw — keeps weights
    of zero, and predicts the channel mean.
    """
    features = gram.shape[-1]
    ridge = penalty * np.eye(features)
    ridge[-1, -1] = 0.0
    coefficients = np.zeros(moment.shape)
    for system in range(gram.shape[0]):
        if gram[system, -1, -1] > 0.0:
            coefficients[system] = np.linalg.solve(gram[system] + ridge, moment[system])
    return coefficients
