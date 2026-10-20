from torch import Tensor

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
from emblema.pretraining.adapters.objective.reconstruction_decoder import ReconstructionDecoder
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class MaskedReconstruction(PretrainingObjective):
    """The first objective: hide part of a window, encode the rest, predict the hidden values.

    A hidden token is marked as padding for the encoder's call, which for a set encoder is the same
    as never observed, so the encoder meets no placeholder it would not meet at inference. The
    decoder puts a placeholder at every hidden position, labelled with the encoder's own channel and
    time encodings, so a channel hidden whole still trains its embedding. The prediction covers
    every position and ``ReconstructionLoss`` decides which count; the masks come from outside, so
    tests and diagnostics can hold them fixed.

    Encoding and decoding are two calls, so an objective that adds a term over the encoder's
    states reads them once and decodes from the same states.

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

    def encoded(self, batch: TokenTensors, padding_mask: Tensor) -> Tensor:
        """The encoder's states of every position, the tokens under ``padding_mask`` unseen."""
        states: Tensor = self.encoder(
            batch.features, batch.channel_ids, batch.timestamps, batch.timeless, padding_mask
        )
        return states

    def decoded(self, states: Tensor, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The value predicted at every position from the states of the tokens the encoder saw."""
        encodings = self.encoder.channel_embedding(batch.channel_ids) + self.encoder.time_encoding(
            batch.timestamps, batch.timeless
        )
        prediction: Tensor = self.decoder(states, masks.hidden, encodings, batch.padding_mask)
        return prediction

    def forward(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        return self.decoded(self.encoded(batch, batch.padding_mask | masks.hidden), batch, masks)

    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The error against the hidden values, and what predicting the channel mean would cost."""
        scorer = ReconstructionLoss(loss)
        error, tokens = scorer.summed(self(batch, masks), batch, masks.hidden)
        trivial, _ = scorer.summed_over_mean(batch, masks.hidden)
        return ObjectiveScore(error=error, trivial=trivial.detach(), tokens=tokens)
