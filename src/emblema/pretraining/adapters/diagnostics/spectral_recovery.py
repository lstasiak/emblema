from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# A least-squares fit with this many observations per coefficient is a spectrum; with fewer it is
# a curve drawn through the points.
OBSERVATIONS_PER_COEFFICIENT = 2


@dataclass(frozen=True)
class SpectralRecovery:
    """Which frequencies of a channel hidden whole the model gives back, and which it loses.

    Each channel hidden whole is fitted by least squares on sines and cosines at one to ``cycles``
    cycles per window, once for the truth and once for the residual of the prediction, and the
    energy is summed per frequency; ``recovered`` is the share of the truth's energy the residual no
    longer holds. A model recovering only the low frequencies has learnt that channels are smooth,
    not how they move.

    A constant and a linear trend absorb what is slower than a cycle and are not reported; cycles
    are whole, as a half-wave over the window is nearly a constant. The fit uses the tokens' own
    instants, so it serves irregular layouts too, where the shares are approximate. A channel with
    fewer tokens than coefficients is skipped and counted.

    Attributes:
        cycles: Highest frequency fitted, in cycles per window.
        truth_energy: Energy of the true values per frequency, summed over channel-windows.
        residual_energy: Energy of the residual per frequency, likewise.
        fitted: Channel-windows the tallies cover.
        skipped: Channel-windows with too few tokens to fit.
    """

    cycles: int
    truth_energy: tuple[float, ...]
    residual_energy: tuple[float, ...]
    fitted: int
    skipped: int

    def __post_init__(self) -> None:
        if self.cycles < 1:
            raise ValueError(f"cycles must be positive, got {self.cycles}")
        if len(self.truth_energy) != self.cycles or len(self.residual_energy) != self.cycles:
            raise ValueError("the energies hold one entry per frequency")

    @classmethod
    def up_to(cls, cycles: int) -> Self:
        """Nothing observed yet, frequencies up to ``cycles`` cycles per window."""
        zeros = (0.0,) * cycles
        return cls(cycles, zeros, zeros, fitted=0, skipped=0)

    @staticmethod
    def cycles_fitting(tokens: int) -> int:
        """The most cycles a channel of ``tokens`` tokens can be fitted up to; zero if none."""
        return max((tokens // OBSERVATIONS_PER_COEFFICIENT - 2) // 2, 0)

    @property
    def coefficients(self) -> int:
        return 2 + 2 * self.cycles

    def observe(self, batch: TokenTensors, masks: TokenMasks, prediction: Tensor) -> Self:
        """The tallies with every channel hidden whole in ``batch`` fitted in."""
        arrays = WindowArrays.of(batch)
        whole = masks.channel.cpu().numpy() & arrays.observed & ~batch.timeless.cpu().numpy()
        # Moved, then widened, as ``TokenTensors.to`` does it: a device with no double precision
        # cannot convert to it, and asking for both in one call gives back zeros.
        predicted = prediction.detach().to("cpu").to(torch.float64).numpy()
        truth = np.array(self.truth_energy)
        residual = np.array(self.residual_energy)
        fitted, skipped = self.fitted, self.skipped
        for row in range(arrays.rows):
            for channel in np.unique(arrays.channel_ids[row, whole[row]]):
                tokens = whole[row] & (arrays.channel_ids[row] == channel)
                if tokens.sum() < OBSERVATIONS_PER_COEFFICIENT * self.coefficients:
                    skipped += 1
                    continue
                basis = self._basis(arrays.times[row, tokens])
                truth += self._energy(basis, arrays.values[row, tokens])
                residual += self._energy(basis, arrays.values[row, tokens] - predicted[row, tokens])
                fitted += 1
        return replace(
            self,
            truth_energy=tuple(truth.tolist()),
            residual_energy=tuple(residual.tolist()),
            fitted=fitted,
            skipped=skipped,
        )

    def recovered(self) -> tuple[float, ...]:
        """Per frequency, the share of the truth's energy the prediction accounts for.

        One where the residual holds none of it, zero where the prediction gave none of it back,
        negative where the prediction added energy the truth did not have.
        """
        return tuple(
            1.0 - residual / truth if truth > 0.0 else 0.0
            for truth, residual in zip(self.truth_energy, self.residual_energy, strict=True)
        )

    def _basis(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        angles = 2.0 * np.pi * np.outer(times, np.arange(1, self.cycles + 1))
        trend = np.column_stack([np.ones(len(times)), times - 0.5])
        return np.concatenate([trend, np.sin(angles), np.cos(angles)], axis=1)

    def _energy(
        self, basis: NDArray[np.float64], signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        coefficients = np.linalg.lstsq(basis, signal, rcond=None)[0]
        sines = coefficients[2 : 2 + self.cycles]
        cosines = coefficients[2 + self.cycles :]
        result: NDArray[np.float64] = sines**2 + cosines**2
        return result
