import torch
from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.encoder_block import EncoderBlock
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture


class ReconstructionDecoder(nn.Module):
    """Predicts the value of every token of a window from the states of the tokens it can see.

    The encoder only ever saw the visible tokens; here the hidden ones re-enter as one learned
    vector each, told apart solely by the channel and time encodings added to every position, and
    a few blocks of the same kind the encoder is made of let them read the visible states. The
    head is a single linear map to the value, or to as many outputs as a token's target has where
    a token is a cell with a presence, a count and five summaries. Everything here is discarded
    after pretraining: the backbone is the encoder, and nothing the encoder exports passes through
    this module.

    The blocks attend with the batch's own padding mask, not the encoder's: a hidden token is a
    token the encoder pretended was absent, but it is present to the decoder, which has to
    predict it.
    """

    def __init__(
        self,
        architecture: EncoderArchitecture,
        *,
        layers: int,
        dropout: float = 0.0,
        outputs: int = 1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(
                f"the decoder needs at least one block to reach the visible tokens, got {layers}"
            )
        if outputs < 1:
            raise ValueError(f"the head predicts at least one output per token, got {outputs}")
        self.mask_token = nn.Parameter(torch.empty(architecture.width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(EncoderBlock(architecture, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(architecture.width)
        self.head = nn.Linear(architecture.width, outputs)

    @property
    def outputs(self) -> int:
        return int(self.head.out_features)

    def forward(
        self, states: Tensor, hidden: Tensor, encodings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        return self.valued(self.streamed(states, hidden, encodings, padding_mask))

    def streamed(
        self, states: Tensor, hidden: Tensor, encodings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """The normalised state at every position before the head, for a term that reads it."""
        placeholder = self.mask_token.to(states.dtype).expand_as(states)
        stream = torch.where(hidden.unsqueeze(-1), placeholder, states) + encodings
        for block in self.blocks:
            stream = block(stream, padding_mask)
        normalised: Tensor = self.norm(stream)
        return normalised

    def valued(self, streamed: Tensor) -> Tensor:
        """What the head reads off each position's normalised state.

        One value per position, ``[batch, tokens]``, under a head of one output; the outputs
        side by side, ``[batch, tokens, outputs]``, under a wider one.
        """
        value: Tensor = self.head(streamed)
        return value.squeeze(-1) if self.outputs == 1 else value
