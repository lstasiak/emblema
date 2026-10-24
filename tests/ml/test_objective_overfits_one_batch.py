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
far below is the cost of answering every hidden token with the window's visible mean. The
objective that also contrasts the halves of the windows is held to both claims at once: the
values recovered as before, and the four windows' halves paired far below the cost of guessing.
"""

import math

import pytest
import torch

from emblema.pretraining.adapters.objective.contrastive_reconstruction import (
    CONTRAST,
    ContrastiveReconstruction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_score import RECONSTRUCTION, ObjectiveScore
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm
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
NO_CONTRAST = ContrastiveTerm.none()
CONTRASTED = ContrastiveTerm.of(weight=0.5, temperature=0.2)
# The control windows are laid every twelve time units and span thirty-two, so consecutive
# windows of a unit share most of their observations; every third is clear of the others.
CONSECUTIVE = slice(0, 2)
APART = slice(0, 6, 3)


def driven(
    control: Control,
    target: ObjectiveTarget,
    contrast: ContrastiveTerm = NO_CONTRAST,
    *,
    windows: slice = CONSECUTIVE,
) -> tuple[list[float], list[float], list[ObjectiveScore]]:
    """The loss and the trivial loss over the steps, each as a mean over the scored tokens."""
    batch = TokenTensors.from_windows(
        control.windows_of(0)[windows] + control.windows_of(1)[windows]
    )
    masks = TokenMasking(MIXTURE).draw(batch, torch.Generator().manual_seed(1))
    assert all(masks.of_kind(kind).any() for kind in MaskKind if MIXTURE.draws(kind))
    torch.manual_seed(1)
    model = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=control.vocabulary_size,
        decoder_layers=1,
        target=target,
        contrast=contrast,
    ).build()
    optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-2)

    losses, trivial, scores = [], [], []
    for step in range(STEPS):
        optimiser.zero_grad()
        score = model.score(batch, masks, SQUARED)
        (score.error / score.tokens).backward()
        optimiser.step()
        model.after_step(step + 1, STEPS)
        losses.append(float(score.error.detach() / score.tokens))
        trivial.append(float(score.trivial / score.tokens))
        scores.append(score)
    return losses, trivial, scores


def test_one_batch_of_control_windows_is_driven_to_nearly_zero_loss(control: Control) -> None:
    losses, trivial, _ = driven(control, ObjectiveTarget.value())

    # The values are normalised per channel, so the loss of an ignorant model is about one.
    assert losses[0] > 0.3
    assert all(t == pytest.approx(trivial[0]) for t in trivial)
    assert min(losses[-10:]) < NEARLY_ZERO, losses[::30]


def test_one_batch_of_control_windows_is_driven_far_below_the_visible_mean_s_cost(
    control: Control,
) -> None:
    losses, trivial, _ = driven(control, STILL)

    # The teacher stands still, so the trivial predictor's cost is one fixed number.
    assert all(t == pytest.approx(trivial[0], rel=1e-5) for t in trivial)
    assert losses[0] > FAR_BELOW_TRIVIAL * trivial[0]
    assert min(losses[-10:]) < FAR_BELOW_TRIVIAL * trivial[0], (losses[::30], trivial[0])


def test_one_batch_of_control_windows_is_reconstructed_and_its_halves_paired_far_below_chance(
    control: Control,
) -> None:
    # Windows that overlap in time are one window to the contrast, which is the point of it;
    # the claim here is that four windows that share nothing are told apart.
    losses, trivial, scores = driven(control, ObjectiveTarget.value(), CONTRASTED, windows=APART)
    reconstruction = [float(s.terms[RECONSTRUCTION] / s.tokens) for s in scores]
    contrast = [float(s.terms[CONTRAST] / s.tokens) for s in scores]
    chance = [float(s.trivial_terms[CONTRAST] / s.tokens) for s in scores]

    # Four windows, so each anchor picks among seven; the chance cost is fixed by the batch.
    assert all(c == pytest.approx(chance[0]) for c in chance)
    assert chance[0] == pytest.approx(CONTRASTED.weight * math.log(7))
    assert contrast[0] > FAR_BELOW_TRIVIAL * chance[0]
    assert min(contrast[-10:]) < FAR_BELOW_TRIVIAL * chance[0], (contrast[::30], chance[0])
    assert min(reconstruction[-10:]) < NEARLY_ZERO, reconstruction[::30]
    assert losses[-1] == pytest.approx(reconstruction[-1] + contrast[-1], rel=1e-5)
    assert all(t == pytest.approx(trivial[0]) for t in trivial)


def test_the_objective_that_predicts_values_is_still_the_one_the_shape_builds_by_default(
    control: Control,
) -> None:
    built = ObjectiveShape(
        architecture=SMALL,
        vocabulary_size=control.vocabulary_size,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
    ).build()

    assert type(built) is MaskedReconstruction
    assert not isinstance(built, ContrastiveReconstruction)


def test_one_batch_of_control_windows_laid_out_as_cells_is_driven_to_nearly_zero_loss(
    control: Control,
) -> None:
    """The objective over cells memorises a fixed batch: the summaries, the presence and the count.

    The cells are laid out twice, from every reading and from the readings the masks leave
    visible, so the claim covers the layout as well as the gradient: a hidden cell's summaries
    are recovered from the cells around it with nothing of the cell itself in them.
    """
    from dataclasses import replace

    from emblema.pretraining.adapters.objective.cell_reconstruction import COUNT, PRESENCE
    from emblema.pretraining.adapters.training.cell_batches import CellBatches
    from emblema.pretraining.domain.training.observation_term import ObservationTerm
    from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens

    layout = CellBatches(BinnedTokens(4))
    collated = layout.collate(
        control.windows_of(0)[CONSECUTIVE] + control.windows_of(1)[CONSECUTIVE]
    )
    masks = TokenMasking(MIXTURE).draw(collated, torch.Generator().manual_seed(1))
    batch = layout.scored(collated, masks)
    assert all(masks.of_kind(kind).any() for kind in MaskKind if MIXTURE.draws(kind))
    torch.manual_seed(1)
    model = ObjectiveShape(
        architecture=replace(SMALL, bins=4),
        vocabulary_size=control.vocabulary_size,
        decoder_layers=1,
        target=ObjectiveTarget.value(),
        observation=ObservationTerm.of(presence_weight=1.0, count_weight=1.0),
    ).build()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    scores = []
    for _ in range(STEPS):
        optimiser.zero_grad()
        score = model.score(batch, masks, SQUARED)
        (score.error / score.tokens).backward()
        optimiser.step()
        scores.append(score)
    reconstruction = [float(s.terms[RECONSTRUCTION] / s.tokens) for s in scores]
    presence = [float(s.terms[PRESENCE] / s.tokens) for s in scores]
    count = [float(s.terms[COUNT] / s.tokens) for s in scores]

    # The control windows are dense, so every cell holds readings: the presence of a hidden
    # cell is a certainty the model has to learn, and its count is nearly constant.
    assert reconstruction[0] > 0.3
    assert presence[0] > 0.1
    assert min(reconstruction[-10:]) < NEARLY_ZERO, reconstruction[::30]
    assert min(presence[-10:]) < NEARLY_ZERO, presence[::30]
    assert min(count[-10:]) < NEARLY_ZERO, count[::30]
    trivial = [float(s.trivial / s.tokens) for s in scores]
    assert all(t == pytest.approx(trivial[0]) for t in trivial)
