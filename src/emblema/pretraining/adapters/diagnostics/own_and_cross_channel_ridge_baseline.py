from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
from emblema.pretraining.adapters.diagnostics.ridge_regression import (
    nearest_values,
    solved,
    validate,
)
from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# The kinds of mask that leave a token's own channel something to interpolate from. A channel
# hidden whole has nothing of its own left, and its strongest linear answer is the cross-channel
# regression alone.
KINDS = (MaskKind.BLOCK, MaskKind.TOKEN)


@dataclass(frozen=True)
class OwnAndCrossChannelRidgeBaseline:
    """The strongest linear answer for a token whose channel still shows something in the window.

    One ridge regression over both sources the matched baselines read apart — the other channels'
    nearest visible values and the channel's own interpolated line — so that a model beating both
    separately is shown to beat linear algebra on the same inputs too. It stands beside the matched
    baselines, which the plan names, rather than replacing them.

    One regression per kind of mask and channel: the kind decides how far the visible neighbours
    lie, and it is read off the masks, known before the value is. Fitted under drawn masks on
    training windows, on hidden tokens of the kind only, as a visible token would be interpolated
    from itself. A token with no visible token of its channel, or a timeless one, gets zero for the
    line.

    Attributes:
        coefficients: Per kind of mask, ``[entries, entries + 2]`` — a row of weights per target
            channel: one per vocabulary entry with the channel's own at zero, then the channel's
            own interpolated value, then the bias. A channel the fit never saw under a kind has a
            row of zeros and predicts the channel mean.
    """

    coefficients: Mapping[MaskKind, NDArray[np.float64]]

    @classmethod
    def fitted(
        cls,
        batches: Iterable[tuple[TokenTensors, TokenMasks]],
        *,
        vocabulary_size: int,
        penalty: float = 1.0,
    ) -> Self:
        """Fit one regression per kind of mask and channel over the hidden tokens of ``batches``.

        Args:
            batches: Training windows, each batch with the masks drawn over it.
            vocabulary_size: Entries of the channel vocabulary.
            penalty: Weight of the ridge term, on every coefficient but the bias.

        Raises:
            ValueError: If ``vocabulary_size`` is not positive or ``penalty`` is negative.
        """
        validate(vocabulary_size, penalty)
        entries = vocabulary_size + 1
        gram = {kind: np.zeros((entries, entries + 2, entries + 2)) for kind in KINDS}
        moment = {kind: np.zeros((entries, entries + 2)) for kind in KINDS}
        for batch, masks in batches:
            arrays = WindowArrays.of(batch)
            for kind, row, targets, rows in _rows(arrays, batch, masks, entries):
                channels, values = arrays.channel_ids[row, targets], arrays.values[row, targets]
                for channel in np.unique(channels):
                    chosen = channels == channel
                    gram[kind][channel] += rows[chosen].T @ rows[chosen]
                    moment[kind][channel] += rows[chosen].T @ values[chosen]
        return cls({kind: solved(gram[kind], moment[kind], penalty) for kind in KINDS})

    def predict(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The baseline's value at every hidden token of a block or a single draw, zero elsewhere.

        A token of a channel hidden whole is left at zero: the cross-channel baseline answers it.
        """
        arrays = WindowArrays.of(batch)
        entries = self.coefficients[KINDS[0]].shape[0]
        prediction = np.zeros_like(arrays.values)
        for kind, row, targets, rows in _rows(arrays, batch, masks, entries):
            weights = self.coefficients[kind][arrays.channel_ids[row, targets]]
            prediction[row, targets] = np.einsum("tf,tf->t", rows, weights)
        return torch.from_numpy(prediction).to(batch.features.device, batch.features.dtype)


def _rows(
    arrays: WindowArrays, batch: TokenTensors, masks: TokenMasks, entries: int
) -> Iterable[tuple[MaskKind, int, NDArray[np.int64], NDArray[np.float64]]]:
    """Per kind and window holding tokens of it: their positions and their design rows."""
    hidden = masks.hidden.cpu().numpy() & arrays.observed
    timeless = batch.timeless.cpu().numpy()
    for kind in KINDS:
        of_kind = masks.of_kind(kind).cpu().numpy() & arrays.observed
        for position in np.flatnonzero(of_kind.any(axis=1)):
            row = int(position)
            targets = np.flatnonzero(of_kind[row])
            series = arrays.series(row, ~hidden[row])
            channels = arrays.channel_ids[row, targets]
            times = arrays.times[row, targets]
            rows = np.zeros((len(targets), entries + 2))
            rows[:, :entries] = nearest_values(series, times, entries)[:, :entries]
            rows[np.arange(len(targets)), channels] = 0.0
            rows[:, entries] = _own_line(series, channels, times, timeless[row, targets])
            rows[:, -1] = 1.0
            yield kind, row, targets, rows


def _own_line(
    series: Mapping[int, ChannelSeries],
    channels: NDArray[np.int64],
    times: NDArray[np.float64],
    timeless: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Each token's own channel interpolated at its instant; zero where there is no line to draw."""
    line = np.zeros(len(channels))
    for channel in np.unique(channels):
        tokens = (channels == channel) & ~timeless
        if int(channel) in series and tokens.any():
            line[tokens] = series[int(channel)].interpolate(times[tokens])
    return line
