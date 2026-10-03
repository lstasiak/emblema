import torch
from torch import Tensor, nn

from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling


class StaticsApartPooling(nn.Module):
    """The readings pooled by a scheme and the static features by their mean, side by side.

    A stay holds a handful of static features against hundreds of readings, so a pooling over
    both weighs each static feature as one reading. Here the scheme sees the static features as
    padding and pools the readings alone, the static features are averaged on their own, and the
    two states are concatenated: the head reads twice the encoder's width, half of it the static
    features. Both halves weigh padding out rather than slicing it out, so the token count stays
    the one dynamic axis and the module traces for export. A window without static features, or
    without readings, gives zeros on that side, whatever the scheme makes of a window it sees as
    nothing but padding: attention would otherwise hand back the mean of every position, the
    static features and the padding among them.

    Attributes:
        readings: The scheme the readings are pooled by.
    """

    def __init__(self, readings: nn.Module) -> None:
        super().__init__()
        self.readings = readings
        self.statics = MaskedMeanPooling()

    def forward(
        self, states: Tensor, padding_mask: Tensor, timestamps: Tensor, timeless: Tensor
    ) -> Tensor:
        unread = padding_mask | timeless
        readings: Tensor = self.readings(states, unread, timestamps, timeless)
        read = (~unread).any(dim=1, keepdim=True).to(readings.dtype)
        statics: Tensor = self.statics(states, padding_mask | ~timeless)
        return torch.cat((readings * read, statics), dim=-1)
