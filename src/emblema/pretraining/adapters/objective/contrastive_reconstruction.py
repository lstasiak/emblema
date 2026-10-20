from torch import Tensor

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.contrast_projection import ContrastProjection
from emblema.pretraining.adapters.objective.contrastive_loss import ContrastiveLoss
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

RECONSTRUCTION = "reconstruction"
CONTRAST = "contrast"


class ContrastiveReconstruction(MaskedReconstruction):
    """Masked reconstruction with a second term: the two halves of a window told apart from others'.

    The masks cut every window in two. The tokens the encoder sees are one view and the tokens
    hidden from it are the other, so the second view is the first's complement, drawn by the same
    draw and no other: the reconstruction predicts the values of the second view from the first,
    and the contrast asks that each view, summarised to one point, pick the other view of its
    window out of every view of every other window in the batch. Two disjoint sets of
    observations of one stay or one month agree on what the window is, and a line through a
    token's neighbours does not answer that.

    Each view goes through the encoder as the reconstruction's view does, the other view's
    tokens marked as padding; the first pass is the reconstruction's, so the decoder predicts
    from the same states the contrast pools. The projection is thrown away with the decoder.
    A window whose masks hid nothing of it has no second view and leaves the contrast, as a
    distractor too; it is still reconstructed, which is nothing. The encoder and the decoder are
    built before the projection, so a run under this objective draws the same encoder and the
    same decoder from its seed as the run that predicts values alone, and differs from it in the
    term only.

    The contrast is weighed against the hidden tokens' loss as the term states: an anchor's mean
    loss, times the weight, times the hidden tokens scored, so that once the batch is divided by
    its tokens the loss is the mean reconstruction plus the weight times the mean contrast, and
    a micro-batch weighs by its tokens as it does under the reconstruction alone. The trivial
    predictor is the reconstruction's beside the contrast's chance, scaled the same way.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
        decoder: The part that predicts the values; thrown away.
        projection: The part that carries a view's pooled state to the contrast; thrown away.
        term: The contrast's weight and temperature.
    """

    def __init__(
        self,
        encoder: SetEncoder,
        *,
        decoder_layers: int,
        contrast: ContrastiveTerm,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(encoder, decoder_layers=decoder_layers, dropout=dropout)
        if not contrast.is_stated:
            raise ValueError("this objective contrasts the views of a window, which needs a term")
        self.projection = ContrastProjection(encoder.architecture)
        self.term = contrast

    @staticmethod
    def views(batch: TokenTensors, masks: TokenMasks) -> tuple[Tensor, Tensor, Tensor]:
        """The padding of each view and which windows are pairs: ``(first, second, valid)``.

        The first view's padding is the reconstruction's: the batch's own and the hidden tokens.
        The second's is everything else, so it holds exactly the hidden tokens that were
        observed. A window is a pair where each view holds a token.
        """
        hidden = masks.hidden & ~batch.padding_mask
        first = batch.padding_mask | hidden
        second = ~hidden
        valid = (~first).any(dim=1) & hidden.any(dim=1)
        return first, second, valid

    def contrasted(self, batch: TokenTensors, masks: TokenMasks) -> tuple[Tensor, Tensor]:
        """The projections of both views, ``[batch, width]`` each, read without the decoder."""
        first, second, _ = self.views(batch, masks)
        return (
            self.projection(self.encoded(batch, first), first),
            self.projection(self.encoded(batch, second), second),
        )

    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The reconstruction's error and the weighed contrast, each beside the trivial cost."""
        first, second, valid = self.views(batch, masks)
        states = self.encoded(batch, first)
        scorer = ReconstructionLoss(loss)
        prediction = self.decoded(states, batch, masks)
        reconstruction, tokens = scorer.summed(prediction, batch, masks.hidden)
        reconstruction_trivial, _ = scorer.summed_over_mean(batch, masks.hidden)

        contrasting = ContrastiveLoss(self.term)
        contrast_sum, anchors = contrasting.summed(
            self.projection(states, first),
            self.projection(self.encoded(batch, second), second),
            valid,
        )
        scale = self.term.weight * tokens.to(contrast_sum.dtype) / anchors.clamp(min=1)
        contrast = contrast_sum * scale
        contrast_trivial = contrasting.chance(valid) * scale

        return ObjectiveScore(
            error=reconstruction + contrast,
            trivial=(reconstruction_trivial + contrast_trivial).detach(),
            tokens=tokens,
            terms={RECONSTRUCTION: reconstruction.detach(), CONTRAST: contrast.detach()},
            trivial_terms={
                RECONSTRUCTION: reconstruction_trivial.detach(),
                CONTRAST: contrast_trivial.detach(),
            },
        )
