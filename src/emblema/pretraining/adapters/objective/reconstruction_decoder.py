import torch
from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.encoder_block import EncoderBlock
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture


class ReconstructionDecoder(nn.Module):
    """Predicts the value of every token of a window from the states of the tokens it can see.

    The encoder only ever saw the visible tokens; here the hidden ones re-enter as one learned
    vector each, told apart solely by the channel and time encodings added to every position, and
    a few blocks of the same kind the encoder is made of let them read the visible states. The
    head is a single linear map to the value. Everything here is discarded after pretraining: the
    backbone is the encoder, and nothing the encoder exports passes through this module.

    The blocks attend with the batch's own padding mask, not the encoder's: a hidden token is a
    token the encoder pretended was absent, but it is present to the decoder, which has to
    predict it.
    """

    def __init__(
        self, architecture: EncoderArchitecture, *, layers: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(
                f"the decoder needs at least one block to reach the visible tokens, got {layers}"
            )
        self.mask_token = nn.Parameter(torch.empty(architecture.width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(EncoderBlock(architecture, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(architecture.width)
        self.head = nn.Linear(architecture.width, 1)

    def forward(
        self, states: Tensor, hidden: Tensor, encodings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        placeholder = self.mask_token.to(states.dtype).expand_as(states)
        stream = torch.where(hidden.unsqueeze(-1), placeholder, states) + encodings
        for block in self.blocks:
            stream = block(stream, padding_mask)
        return self.head(self.norm(stream)).squeeze(-1)
