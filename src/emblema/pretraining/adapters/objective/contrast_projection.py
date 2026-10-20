from torch import Tensor, nn

from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling


class ContrastProjection(nn.Module):
    """One point per window from the states of the tokens a view holds, for the contrast to compare.

    The states of the view's tokens are averaged into one embedding, as the task's head and the
    probe pool a backbone's states, and two linear maps with the encoder's nonlinearity between
    them carry that embedding to where the similarities are read. The maps are what the
    contrast is allowed to shape without shaping the states themselves (Chen et al. 2020): the
    backbone is the encoder, the pooled state is what a task reads, and this module is thrown
    away with the decoder.
    """

    def __init__(self, architecture: EncoderArchitecture) -> None:
        super().__init__()
        self.pooling = MaskedMeanPooling()
        self.layers = nn.Sequential(
            nn.Linear(architecture.width, architecture.width),
            nn.GELU(),
            nn.Linear(architecture.width, architecture.width),
        )

    def forward(self, states: Tensor, padding_mask: Tensor) -> Tensor:
        """The projected mean state of the tokens not under ``padding_mask``: ``[batch, width]``."""
        projected: Tensor = self.layers(self.pooling(states, padding_mask))
        return projected
