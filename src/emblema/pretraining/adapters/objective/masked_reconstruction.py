from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.reconstruction_decoder import ReconstructionDecoder
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class MaskedReconstruction(nn.Module):
    """The self-supervised objective: hide part of a window, encode the rest, predict the hidden.

    A hidden token leaves the encoder's input altogether — it is marked as padding for that call,
    which for a set encoder is the same as never having been observed — so the encoder never meets
    a placeholder it will not meet at inference, and its attention pays only for the tokens it can
    see. The decoder then puts a placeholder at every hidden position, labels each with the channel
    and time encodings the encoder itself uses, and reads the visible states to predict the value.
    Sharing those two input modules means a channel hidden whole still teaches its own embedding
    through the question the decoder asks about it.

    The prediction covers every position; ``ReconstructionLoss`` decides which ones count. The
    masks come from outside, so a test can hold them fixed and a diagnostic can hand the same masks
    to a baseline.

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
