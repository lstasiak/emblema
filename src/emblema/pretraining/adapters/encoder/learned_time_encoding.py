import torch
from torch import Tensor, nn


class LearnedTimeEncoding(nn.Module):
    """A token's position in its window as sines and cosines whose frequencies and phases learn.

    The fixed encoding doubles its frequencies from half a cycle per window and shifts none of
    them; here each frequency and each phase learns, as in the learnt time embedding of networks
    published on irregular clinical series. What learns is the shift of each octave from the
    fixed one and each phase, both zero at the start, so a run under this module begins with the
    encoding every backbone had, and a weight decay pulls the frequencies back towards it rather
    than towards nothing. As in the fixed module the angles are formed in single precision, since
    a wide encoding's top frequency overflows half precision, and a timeless token gets a zero
    encoding through ``where``.

    Attributes:
        octaves: Doublings of half a cycle per window, one per frequency; the fixed encoding's.
        octave_shift: How far each octave has moved from the fixed one; learnt, zero at the start.
        phases: Shift of each frequency's angle, in radians; learnt, zero at the start.
    """

    octaves: Tensor

    def __init__(self, frequencies: int, width: int) -> None:
        super().__init__()
        self.register_buffer("octaves", torch.arange(frequencies, dtype=torch.float32))
        self.octave_shift = nn.Parameter(torch.zeros(frequencies))
        self.phases = nn.Parameter(torch.zeros(frequencies))
        self.projection = nn.Linear(2 * frequencies, width)

    @property
    def frequencies(self) -> Tensor:
        """Angular frequencies, in radians per window, a position is multiplied by."""
        return torch.pi * torch.exp2(self.octaves.float() + self.octave_shift.float())

    def forward(self, timestamps: Tensor, timeless: Tensor) -> Tensor:
        angles = timestamps.unsqueeze(-1).float() * self.frequencies + self.phases.float()
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        encoded = self.projection(features.to(self.projection.weight.dtype))
        return torch.where(timeless.unsqueeze(-1), torch.zeros_like(encoded), encoded)
