import torch
from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.encoder_block import EncoderBlock
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture


class LatentPredictor(nn.Module):
    """Predicts the state of every token of a window from the states of the tokens it can see.

    The same shape as the decoder that predicts a value, with a head as wide as the state: the
    hidden tokens re-enter as one learned vector each, told apart solely by the channel and time
    encodings added to every position, and a few blocks of the encoder's kind let them read the
    visible states. Everything here is discarded after pretraining: the backbone is the encoder,
    and nothing the encoder exports passes through this module. A class of its own rather than
    the decoder with a wider head, so that what either stores under its name stays what it was.

    The blocks attend with the batch's own padding mask, not the encoder's: a hidden token is a
    token the encoder pretended was absent, but it is present to the predictor, which has to
    predict it.
    """

    def __init__(
        self, architecture: EncoderArchitecture, *, layers: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(
                f"the predictor needs at least one block to reach the visible tokens, got {layers}"
            )
        self.mask_token = nn.Parameter(torch.empty(architecture.width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(EncoderBlock(architecture, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(architecture.width)
        self.head = nn.Linear(architecture.width, architecture.width)

    def forward(
        self, states: Tensor, hidden: Tensor, encodings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        placeholder = self.mask_token.to(states.dtype).expand_as(states)
        stream = torch.where(hidden.unsqueeze(-1), placeholder, states) + encodings
        for block in self.blocks:
            stream = block(stream, padding_mask)
        predicted: Tensor = self.head(self.norm(stream))
        return predicted
