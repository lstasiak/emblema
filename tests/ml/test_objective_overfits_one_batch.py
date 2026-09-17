"""The one test the whole project rests on: the objective drives one batch to almost no loss.

A batch of control windows, one fixed draw of masks, a small encoder and a few hundred steps of
Adam. If the loss does not go to nearly zero, something between the tokens and the gradient is
broken — the masks hide what the loss scores, the decoder cannot reach the visible states, the
target is read from the wrong column — and no amount of training on real data would say which.
If it does, the pipeline can at least memorise, and every later failure is a failure to
generalise rather than a failure to learn.

The masks are drawn once and held: a target that moves every step is a different, harder claim,
and the one that belongs here is that a fixed set of hidden values can be recovered from the
visible ones.
"""

import pytest
import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.support.control_corpus import Control
from tests.support.encoders import SMALL

pytestmark = pytest.mark.ml

MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
STEPS = 150
NEARLY_ZERO = 0.01


def test_one_batch_of_control_windows_is_driven_to_nearly_zero_loss(control: Control) -> None:
    batch = TokenTensors.from_windows(control.windows_of(0)[:2] + control.windows_of(1)[:2])
    masks = TokenMasking(MIXTURE).draw(batch, torch.Generator().manual_seed(1))
    torch.manual_seed(1)
    model = MaskedReconstruction(
        SetEncoder.for_vocabulary(SMALL, control.vocabulary_size), decoder_layers=1
    )
    loss = ReconstructionLoss(ObjectiveLoss(kind=LossKind.MSE))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(STEPS):
        optimiser.zero_grad()
        step = loss(model(batch, masks), batch, masks)
        step.backward()
        optimiser.step()
        losses.append(step.item())

    assert all(kind.any() for kind in (masks.of_kind(k) for k in MaskKind))
    # The values are normalised per channel, so the loss of an ignorant model is about one.
    assert losses[0] > 0.3
    assert min(losses[-10:]) < NEARLY_ZERO, losses[::30]
