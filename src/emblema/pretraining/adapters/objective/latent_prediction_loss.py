from torch import Tensor, nn

from emblema.pretraining.adapters.objective.reconstruction_loss import elementwise, losses_dtype
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class LatentPredictionLoss(nn.Module):
    """The error of a predicted state against the teacher's, over the hidden observed tokens.

    The same rule the value objective's scorer applies, for a target that is a vector: a
    position that is padding is not a token and a token that was never hidden was given to the
    model, so neither is scored, whatever the masks claim. One token's loss is the reading
    averaged over the coordinates of its state, so a token weighs the same as it does under the
    value objective and a run's learning rate means the same thing under either.

    The trivial predictor is the mean of the teacher's states over the window's *visible*
    tokens — what a model that read nothing of a hidden token but the rest of the window could
    answer, as the channel mean is for a value. It reads no hidden token's target, and every
    window keeps a visible token, so it is always defined.
    """

    def __init__(self, loss: ObjectiveLoss) -> None:
        """Score the hidden tokens by ``loss``, coordinate by coordinate."""
        super().__init__()
        self.loss = loss

    def of_tokens(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Per position, the reading averaged over the coordinates of the state, in ≥ fp32."""
        dtype = losses_dtype(prediction)
        return elementwise(self.loss, prediction.to(dtype), target.to(dtype)).mean(-1)

    def summed(
        self, prediction: Tensor, target: Tensor, batch: TokenTensors, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The loss over ``positions`` that hold an observed token, and how many were scored."""
        scored = positions & ~batch.padding_mask
        losses = self.of_tokens(prediction, target) * scored.to(losses_dtype(prediction))
        return losses.sum(), scored.sum()

    def summed_over_visible_mean(
        self, target: Tensor, batch: TokenTensors, masks: TokenMasks, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """What predicting the window's visible mean state costs over those positions."""
        return self.summed(self.visible_mean(target, batch, masks), target, batch, positions)

    @staticmethod
    def visible_mean(target: Tensor, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        """The trivial prediction: each window's mean target over its visible tokens, everywhere.

        A window without a visible token has no mean and predicts zero; no window a masking draws
        is one, since every window keeps a visible token.
        """
        visible = (~batch.padding_mask & ~masks.hidden).unsqueeze(-1).to(target.dtype)
        count = visible.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (target * visible).sum(dim=1, keepdim=True) / count
        return mean.expand_as(target)
