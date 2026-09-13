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
        scored = positions & ~batch.padding_mask
        target = batch.features[..., 0].to(prediction.dtype)
        squared = (prediction - target).pow(2) * scored.to(prediction.dtype)
        return squared.sum() / scored.sum().clamp(min=1)
