from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.reconstruction_decoder import ReconstructionDecoder
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class MaskedReconstruction(nn.Module):
    """The self-supervised objective: hide part of a window, encode the rest, predict the hidden.

    A hidden token is marked as padding for the encoder's call, which for a set encoder is the same
    as never observed, so the encoder meets no placeholder it would not meet at inference. The
    decoder puts a placeholder at every hidden position, labelled with the encoder's own channel and
    time encodings, so a channel hidden whole still trains its embedding. The prediction covers
    every position and ``ReconstructionLoss`` decides which count; the masks come from outside, so
    tests and diagnostics can hold them fixed.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
        decoder: The part that is thrown away.
    """

    def __init__(self, encoder: SetEncoder, *, decoder_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = ReconstructionDecoder(
            encoder.architecture, layers=decoder_layers, dropout=dropout
        )

    def forward(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        hidden = masks.hidden
        states = self.encoder(
            batch.features,
            batch.channel_ids,
            batch.timestamps,
            batch.timeless,
            batch.padding_mask | hidden,
        )
        encodings = self.encoder.channel_embedding(batch.channel_ids) + self.encoder.time_encoding(
            batch.timestamps, batch.timeless
        )
        return self.decoder(states, hidden, encodings, batch.padding_mask)
