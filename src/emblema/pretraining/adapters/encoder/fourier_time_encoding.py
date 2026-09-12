import torch
from torch import Tensor, nn


class FourierTimeEncoding(nn.Module):
    """A token's position in its window as sines and cosines at fixed frequencies.

    The frequencies double from half a cycle per window upwards and are not learned: fixed, the
    encoding has nothing to overfit and extrapolates to positions training never showed. The sines
    and cosines are projected to the encoder's width. A timeless token gets a zero encoding rather
    than one at an invented time, chosen with ``where`` so that the module traces for export.

    Attributes:
        frequencies: Angular frequencies, in radians per window, the position is multiplied by.
    """

    frequencies: Tensor

    def __init__(self, frequencies: int, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies", torch.pi * 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        )
        self.projection = nn.Linear(2 * frequencies, width)

    def forward(self, timestamps: Tensor, timeless: Tensor) -> Tensor:
        angles = timestamps.unsqueeze(-1) * self.frequencies
        encoded = self.projection(torch.cat([angles.sin(), angles.cos()], dim=-1))
        return torch.where(timeless.unsqueeze(-1), torch.zeros_like(encoded), encoded)
