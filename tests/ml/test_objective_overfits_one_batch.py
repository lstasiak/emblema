"""The one test the whole project rests on: the objective drives one batch to almost no loss.

A batch of control windows, one fixed draw of masks, a small encoder and a few hundred steps of
Adam. If the loss does not go to nearly zero, something between the tokens and the gradient is
broken — the masks hide what the loss scores, the decoder cannot reach the visible states, the
target is read from the wrong column — and no amount of training on real data would say which.
If it does, the pipeline can at least memorise, and every later failure is a failure to
generalise rather than a failure to learn.

The masks are drawn once and held: a target that moves every step is a different, harder claim,
and the one that belongs here is that a fixed set of hidden values can be recovered from the
visible ones. The objective that predicts states is held to the same claim with its teacher held
still, so its targets are fixed too; they are states rather than one value, so what it is driven
far below is the cost of answering every hidden token with the window's visible mean.
"""

import pytest
import torch

from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.support.control_corpus import Control
from tests.support.encoders import SMALL

pytestmark = pytest.mark.ml

MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
SQUARED = ObjectiveLoss(kind=LossKind.MSE)
STEPS = 150
NEARLY_ZERO = 0.01
# A state is not one value: the predictor is held to a tenth of what the visible mean costs.
FAR_BELOW_TRIVIAL = 0.1
STILL = ObjectiveTarget.latent(layers=SMALL.layers, momentum=1.0, final_momentum=1.0)


def driven(control: Control, target: ObjectiveTarget) -> tuple[list[float], list[float]]:
    """The loss and the trivial loss over the steps, each as a mean over the scored tokens."""
    batch = TokenTensors.from_windows(control.windows_of(0)[:2] + control.windows_of(1)[:2])
    masks = TokenMasking(MIXTURE).draw(batch, torch.Generator().manual_seed(1))
    assert all(masks.of_kind(kind).any() for kind in MaskKind if MIXTURE.draws(kind))
    torch.manual_seed(1)
    model = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=control.vocabulary_size,
        decoder_layers=1,
        target=target,
    ).build()
    optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-2)

    losses, trivial = [], []
    for step in range(STEPS):
        optimiser.zero_grad()
        score = model.score(batch, masks, SQUARED)
        (score.error / score.tokens).backward()
        optimiser.step()
        model.after_step(step + 1, STEPS)
        losses.append(float(score.error.detach() / score.tokens))
        trivial.append(float(score.trivial / score.tokens))
    return losses, trivial


def test_one_batch_of_control_windows_is_driven_to_nearly_zero_loss(control: Control) -> None:
    losses, trivial = driven(control, ObjectiveTarget.value())

    # The values are normalised per channel, so the loss of an ignorant model is about one.
    assert losses[0] > 0.3
    assert all(t == pytest.approx(trivial[0]) for t in trivial)
    assert min(losses[-10:]) < NEARLY_ZERO, losses[::30]


def test_one_batch_of_control_windows_is_driven_far_below_the_visible_mean_s_cost(
    control: Control,
) -> None:
    losses, trivial = driven(control, STILL)

    # The teacher stands still, so the trivial predictor's cost is one fixed number.
    assert all(t == pytest.approx(trivial[0], rel=1e-5) for t in trivial)
    assert losses[0] > FAR_BELOW_TRIVIAL * trivial[0]
    assert min(losses[-10:]) < FAR_BELOW_TRIVIAL * trivial[0], (losses[::30], trivial[0])


def test_the_objective_that_predicts_values_is_still_the_one_the_shape_builds_by_default(
    control: Control,
) -> None:
    built = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=control.vocabulary_size,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
    ).build()

    assert isinstance(built, MaskedReconstruction)
