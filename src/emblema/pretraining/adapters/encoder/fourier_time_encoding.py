import torch
from torch import Tensor, nn


class FourierTimeEncoding(nn.Module):
    """A token's position in its window as sines and cosines at fixed frequencies.

    The frequencies double from half a cycle per window upwards and are not learned: fixed, the
    encoding has nothing to overfit and extrapolates to positions training never showed. The sines
    and cosines are projected to the encoder's width. A timeless token gets a zero encoding rather
    than one at an invented time, chosen with ``where`` so that the module traces for export.

    What is stored is the octave of each frequency rather than the frequency itself, and the angles
    are formed in single precision whatever precision the module is held in: a wide encoding reaches
    some 10^5 radians per window at its top frequency, which overflows half precision, while the
    octaves are small integers every floating type holds exactly.

    Attributes:
        octaves: Doublings of half a cycle per window, one per frequency.
    """

    octaves: Tensor

    def __init__(self, frequencies: int, width: int) -> None:
        super().__init__()
        self.register_buffer("octaves", torch.arange(frequencies, dtype=torch.float32))
        self.projection = nn.Linear(2 * frequencies, width)

    @property
    def frequencies(self) -> Tensor:
        """Angular frequencies, in radians per window, a position is multiplied by."""
        return torch.pi * torch.exp2(self.octaves.float())

    def forward(self, timestamps: Tensor, timeless: Tensor) -> Tensor:
        angles = timestamps.unsqueeze(-1).float() * self.frequencies
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        encoded = self.projection(features.to(self.projection.weight.dtype))
        return torch.where(timeless.unsqueeze(-1), torch.zeros_like(encoded), encoded)
