from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.summary_loss import SummaryLoss, SummaryUnits
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.domain.training.summary_term import SUMMARIES, SummaryTerm
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

RECONSTRUCTION = "reconstruction"
SUMMARY = "summaries"


class SummaryReconstruction(MaskedReconstruction):
    """Masked reconstruction beside each channel's level and extremes of the hidden part.

    For every channel of a window that lost tokens, the decoder's normalised states at those
    tokens are averaged and a linear map, thrown away with the decoder, answers the mean, minimum
    and maximum of their values. The decoder places every hidden token by its channel and time,
    so a channel hidden whole is answered from the other channels as its tokens are, and the
    gradient reaches the encoder's states as the reconstruction's does. A token's value and its
    channel's summaries are read off the same decoder pass.

    The term is weighed against the hidden tokens' loss as its weight states: the units' mean
    loss, times the weight, times the hidden tokens scored, so that once the batch is divided by
    its tokens the loss is the mean reconstruction plus the weight times the mean summary loss,
    and a micro-batch weighs by its tokens as it does under the reconstruction alone. The trivial
    predictor is the reconstruction's beside the channel's mean for every summary, scaled the
    same way. The encoder and the decoder are built before the head, so a run under this
    objective draws the same encoder and decoder from its seed as the run that predicts values
    alone, and differs from it in the term only.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
        decoder: The part that predicts the values; thrown away.
        summary_head: The map from a channel's pooled decoder states to its summaries; thrown away.
        term: The summaries' weight.
    """

    def __init__(
        self,
        encoder: SetEncoder,
        *,
        decoder_layers: int,
        summary: SummaryTerm,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(encoder, decoder_layers=decoder_layers, dropout=dropout)
        if not summary.is_stated:
            raise ValueError("this objective predicts each channel's summaries, which needs a term")
        self.summary_head = nn.Linear(encoder.architecture.width, len(SUMMARIES))
        self.term = summary

    def summarised(self, batch: TokenTensors, masks: TokenMasks) -> tuple[Tensor, SummaryUnits]:
        """The summaries answered for every unit of the batch, and the units they answer."""
        streamed = self.streamed(batch, masks)
        units = SummaryUnits.of(batch, masks.hidden)
        answered: Tensor = self.summary_head(units.pooled(streamed))
        return answered, units

    def streamed(self, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The decoder's normalised state at every position, from the tokens the encoder saw."""
        states = self.encoded(batch, batch.padding_mask | masks.hidden)
        return self.decoder.streamed(
            states, masks.hidden, self.encodings(batch), batch.padding_mask
        )

    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The reconstruction's error and the weighed summaries, each beside the trivial cost."""
        streamed = self.streamed(batch, masks)
        scorer = ReconstructionLoss(loss)
        reconstruction, tokens = scorer.summed(self.decoder.valued(streamed), batch, masks.hidden)
        reconstruction_trivial, _ = scorer.summed_over_mean(batch, masks.hidden)

        units = SummaryUnits.of(batch, masks.hidden)
        summarising = SummaryLoss(loss)
        summary_sum, scored = summarising.summed(
            self.summary_head(units.pooled(streamed)), batch, units
        )
        scale = self.term.weight * tokens.to(summary_sum.dtype) / scored.clamp(min=1)
        summary = summary_sum * scale
        summary_trivial = summarising.summed_trivial(batch, units) * scale

        return ObjectiveScore(
            error=reconstruction + summary,
            trivial=(reconstruction_trivial + summary_trivial).detach(),
            tokens=tokens,
            terms={RECONSTRUCTION: reconstruction.detach(), SUMMARY: summary.detach()},
            trivial_terms={
                RECONSTRUCTION: reconstruction_trivial.detach(),
                SUMMARY: summary_trivial.detach(),
            },
        )
