from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class WindowArrays:
    """A batch seen as numpy arrays on the host, which is where the trivial baselines work.

    A baseline visits windows and channels one at a time, so it reads the batch through numpy
    rather than through the device the model runs on; the copy is made once per batch here, in
    double precision, so that a baseline's arithmetic is not what limits the comparison.

    Attributes:
        channel_ids: ``[batch, tokens]`` int64.
        times: ``[batch, tokens]`` float64 — position within the window.
        values: ``[batch, tokens]`` float64 — the normalised value, the reconstruction target.
        observed: ``[batch, tokens]`` bool — not padding.
    """

    channel_ids: NDArray[np.int64]
    times: NDArray[np.float64]
    values: NDArray[np.float64]
    observed: NDArray[np.bool_]

    @classmethod
    def of(cls, batch: TokenTensors) -> Self:
        host = batch.to("cpu", torch.float64)
        return cls(
            channel_ids=host.channel_ids.numpy(),
            times=host.timestamps.numpy(),
            values=host.features[..., 0].numpy(),
            observed=~host.padding_mask.numpy(),
        )

    @property
    def rows(self) -> int:
        return int(self.observed.shape[0])

    def series(self, row: int, visible: NDArray[np.bool_]) -> Mapping[int, ChannelSeries]:
        """The series of every channel of window ``row`` built from its ``visible`` tokens."""
        return ChannelSeries.of_window(
            self.channel_ids[row], self.times[row], self.values[row], visible & self.observed[row]
        )
