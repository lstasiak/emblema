from torch import Tensor, nn

from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling


class MeanPooling(nn.Module):
    """The mean over a window's observed states, called as every pooling of a head is called.

    The arithmetic is the shared pooling's, so a candidate under the mean computes and exports
    what it did before the head had a pooling knob; this only takes the times a tail reads and
    leaves them unread.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean = MaskedMeanPooling()

    def forward(
        self, states: Tensor, padding_mask: Tensor, timestamps: Tensor, timeless: Tensor
    ) -> Tensor:
        pooled: Tensor = self.mean(states, padding_mask)
        return pooled
