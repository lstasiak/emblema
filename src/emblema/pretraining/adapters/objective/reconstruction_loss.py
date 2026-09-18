import torch
from torch import Tensor, nn
from torch.nn import functional

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class ReconstructionLoss(nn.Module):
    """The error over the hidden tokens that were observed, and over nothing else.

    A position that is padding is not a token and a token that was never hidden was given to the
    model, so scoring either would lower the loss without the model having predicted anything —
    the mistake that makes an objective look like it works. The rule is applied here, on the
    prediction, whatever the masks claim: masks built by hand are held to it too. The mean is over
    the scored tokens of the whole batch, so a window that hid three tokens does not weigh as much
    as one that hid three hundred.

    What a miss is worth is the run's to state, not this class's: a squared error, or a bounded
    one that stops a single excursion from deciding the gradient. The reading is therefore given
    and never defaulted, and ``ObjectiveLoss.of_error`` is the definition these tensors are held
    to. The target is the normalised value of the token. The same scorer serves a trivial
    baseline, so that a model and the baseline it is measured against are measured the same way;
    ``over`` takes the positions to score, for a diagnostic that scores one kind of mask at a time.
    """

    def __init__(self, loss: ObjectiveLoss) -> None:
        """Score the hidden tokens by ``loss``."""
        super().__init__()
        self.loss = loss

    def forward(self, prediction: Tensor, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        return self.over(prediction, batch, masks.hidden)

    def over(self, prediction: Tensor, batch: TokenTensors, positions: Tensor) -> Tensor:
        """The loss over ``positions`` that hold an observed token; zero where there is none."""
        total, scored = self.summed(prediction, batch, positions)
        return total / scored.clamp(min=1)

    def summed(
        self, prediction: Tensor, batch: TokenTensors, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The loss over those positions before it is divided, and how many were scored.

        What one batch contributes to a larger batch's mean. Whoever sums several batches into one
        gradient adds these and divides once, and then a batch that hid three tokens weighs a
        hundredth of one that hid three hundred — the rule this class applies within a batch,
        applied across them.
        """
        scored = positions & ~batch.padding_mask
        losses = self.of_tokens(prediction, batch) * scored.to(losses_dtype(prediction))
        return losses.sum(), scored.sum()

    def of_tokens(self, prediction: Tensor, batch: TokenTensors) -> Tensor:
        """Per position, what the run's reading makes of the distance from the token's target.

        Nothing is masked here, padding included: this is the one place the target is read, for
        whoever sums the losses over positions of their own choosing. Read in single precision
        at least, whatever the prediction came in: a half-precision sum over one batch of a
        corpus with excursions of hundreds of deviations passes 65,504 and comes back infinite,
        and a target cast to half loses the excursion's value before it is scored. A prediction
        read in double keeps its double.
        """
        dtype = losses_dtype(prediction)
        prediction = prediction.to(dtype)
        target = batch.features[..., 0].to(dtype)
        if self.loss.kind is LossKind.MSE:
            return functional.mse_loss(prediction, target, reduction="none")
        return functional.huber_loss(
            prediction, target, reduction="none", delta=self.loss.huber_delta
        )

    def summed_over_mean(self, batch: TokenTensors, positions: Tensor) -> tuple[Tensor, Tensor]:
        """What predicting the channel mean costs over those positions, and how many were scored.

        The trivial predictor is zero in normalised units, and what a run learnt is read against
        it under the run's own reading: a bounded loss against a variance would be two different
        things divided by each other.
        """
        return self.summed(torch.zeros_like(batch.features[..., 0]), batch, positions)


def losses_dtype(prediction: Tensor) -> torch.dtype:
    """The precision the losses are read in: the prediction's, widened to single at least."""
    return torch.promote_types(prediction.dtype, torch.float32)
