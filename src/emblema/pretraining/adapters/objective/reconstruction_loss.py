from torch import Tensor, nn

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class ReconstructionLoss(nn.Module):
    """Mean squared error over the hidden tokens that were observed, and over nothing else.

    A position that is padding is not a token and a token that was never hidden was given to the
    model, so scoring either would lower the loss without the model having predicted anything —
    the mistake that makes an objective look like it works. The rule is applied here, on the
    prediction, whatever the masks claim: masks built by hand are held to it too. The mean is over
    the scored tokens of the whole batch, so a window that hid three tokens does not weigh as much
    as one that hid three hundred.

    The target is the normalised value of the token. The same scorer serves a trivial baseline, so
    that a model and the baseline it is measured against are measured the same way; ``over``
    takes the positions to score, for a diagnostic that scores one kind of mask at a time.
    """

    def forward(self, prediction: Tensor, batch: TokenTensors, masks: TokenMasks) -> Tensor:
        return self.over(prediction, batch, masks.hidden)

    @staticmethod
    def over(prediction: Tensor, batch: TokenTensors, positions: Tensor) -> Tensor:
        """The error over ``positions`` that hold an observed token; zero where there is none."""
        total, scored = ReconstructionLoss.summed(prediction, batch, positions)
        return total / scored.clamp(min=1)

    @staticmethod
    def summed(prediction: Tensor, batch: TokenTensors, positions: Tensor) -> tuple[Tensor, Tensor]:
        """The error over those positions before it is divided, and how many were scored.

        What one batch contributes to a larger batch's mean. Whoever sums several batches into one
        gradient adds these and divides once, and then a batch that hid three tokens weighs a
        hundredth of one that hid three hundred — the rule this class applies within a batch,
        applied across them.
        """
        scored = positions & ~batch.padding_mask
        squared = ReconstructionLoss.squared_error(prediction, batch) * scored.to(prediction.dtype)
        return squared.sum(), scored.sum()

    @staticmethod
    def squared_error(prediction: Tensor, batch: TokenTensors) -> Tensor:
        """Per position, the squared distance of the prediction from the token's target.

        Nothing is masked here, padding included: this is the one place the target is read, for
        whoever sums the errors over positions of their own choosing.
        """
        target = batch.features[..., 0].to(prediction.dtype)
        return (prediction - target).pow(2)
