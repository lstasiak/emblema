import copy
import math

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402
from torch.nn import functional  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.contrastive_loss import ContrastiveLoss  # noqa: E402
from emblema.pretraining.adapters.objective.contrastive_reconstruction import (  # noqa: E402
    CONTRAST,
    RECONSTRUCTION,
    ContrastiveReconstruction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.contrastive_term import ContrastiveTerm  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
TERM = ContrastiveTerm.of(weight=0.5, temperature=0.2)
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def objective(*, seed: int = 1, term: ContrastiveTerm = TERM) -> ContrastiveReconstruction:
    torch.manual_seed(seed)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)
    return ContrastiveReconstruction(encoder, decoder_layers=1, contrast=term).eval()


def masks_over(batch: TokenTensors, strategy: MaskingStrategy, *, seed: int = 1) -> TokenMasks:
    return TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(seed))


def masks_hiding(hidden: Tensor) -> TokenMasks:
    return TokenMasks(
        channel=torch.zeros_like(hidden),
        block=hidden,
        token=torch.zeros_like(hidden),
        horizon=torch.zeros_like(hidden),
    )


def with_features(batch: TokenTensors, features: Tensor) -> TokenTensors:
    return TokenTensors(
        features, batch.channel_ids, batch.timestamps, batch.timeless, batch.padding_mask
    )


def by_hand(first: Tensor, second: Tensor, temperature: float) -> Tensor:
    """NT-Xent summed over every anchor, written out as the paper writes it."""
    halves = functional.normalize(torch.cat([first, second]), dim=-1)
    pairs = first.shape[0]
    total = torch.zeros(())
    for anchor in range(2 * pairs):
        positive = (anchor + pairs) % (2 * pairs)
        similarities = torch.stack(
            [
                halves[anchor] @ halves[other] / temperature
                for other in range(2 * pairs)
                if other != anchor
            ]
        )
        own = halves[anchor] @ halves[positive] / temperature
        total = total - (own - torch.logsumexp(similarities, dim=0))
    return total


def test_the_prediction_is_the_reconstruction_s_and_the_decoder_is_drawn_as_without_the_term(
    strategy: MaskingStrategy,
) -> None:
    batch = random_batch(3, 40, seed=2, padding=7)
    masks = masks_over(batch, strategy)
    torch.manual_seed(1)
    plain = MaskedReconstruction(
        SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE), decoder_layers=1
    ).eval()

    contrasted = objective(seed=1)

    torch.testing.assert_close(contrasted(batch, masks), plain(batch, masks), rtol=0, atol=0)
    for (name, mine), (_, theirs) in zip(
        [(n, p) for n, p in contrasted.named_parameters() if not n.startswith("projection.")],
        plain.named_parameters(),
        strict=True,
    ):
        assert torch.equal(mine, theirs), name


def test_the_second_view_holds_exactly_the_hidden_tokens_that_were_observed() -> None:
    batch = random_batch(2, 12, seed=3, padding=4)
    hidden = torch.zeros(2, 12, dtype=torch.bool)
    hidden[0, :3] = True
    hidden[1, 2:5] = True
    hidden[1, -2:] = True  # under the padding, which is no token

    first, second, valid = ContrastiveReconstruction.views(batch, masks_hiding(hidden))

    observed = ~batch.padding_mask
    assert torch.equal(~second, hidden & observed)
    assert torch.equal(~first, observed & ~hidden)
    assert torch.equal(~first | ~second, observed)
    assert not (~first & ~second).any()
    assert valid.tolist() == [True, True]


def test_each_view_s_projection_reads_its_own_tokens_and_none_of_the_other_s(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=3, padding=5)
    masks = masks_over(batch, strategy)
    other = random_batch(2, 48, seed=4)
    hidden = masks.hidden.unsqueeze(-1)
    hidden_scrambled = with_features(batch, torch.where(hidden, other.features, batch.features))
    visible_scrambled = with_features(batch, torch.where(hidden, batch.features, other.features))

    first, second = model.contrasted(batch, masks)
    first_under_hidden_scrambled, second_under_hidden_scrambled = model.contrasted(
        hidden_scrambled, masks
    )
    first_under_visible_scrambled, second_under_visible_scrambled = model.contrasted(
        visible_scrambled, masks
    )

    torch.testing.assert_close(first_under_hidden_scrambled, first, **TOLERANCE)
    assert not torch.allclose(second_under_hidden_scrambled, second, **TOLERANCE)
    torch.testing.assert_close(second_under_visible_scrambled, second, **TOLERANCE)
    assert not torch.allclose(first_under_visible_scrambled, first, **TOLERANCE)


def test_the_contrast_is_the_normalised_temperature_cross_entropy_over_both_halves() -> None:
    torch.manual_seed(2)
    first, second = torch.randn(4, 8), torch.randn(4, 8)
    scorer = ContrastiveLoss(TERM)

    summed, anchors = scorer.summed(first, second, torch.ones(4, dtype=torch.bool))

    assert int(anchors) == 8
    torch.testing.assert_close(summed, by_hand(first, second, TERM.temperature))


def test_the_chance_cost_is_the_log_of_the_candidates_per_anchor_and_equal_halves_cost_it() -> None:
    scorer = ContrastiveLoss(TERM)
    valid = torch.ones(5, dtype=torch.bool)
    same = torch.ones(5, 8)

    chance = scorer.chance(valid)
    summed, anchors = scorer.summed(same, same, valid)

    torch.testing.assert_close(chance, torch.tensor(10 * math.log(9)))
    torch.testing.assert_close(summed, chance)
    assert int(anchors) == 10


def test_a_window_that_is_no_pair_leaves_the_contrast_as_an_anchor_and_as_a_distractor() -> None:
    torch.manual_seed(3)
    first, second = torch.randn(4, 8), torch.randn(4, 8)
    scorer = ContrastiveLoss(TERM)
    valid = torch.tensor([True, False, True, True])

    summed, anchors = scorer.summed(first, second, valid)

    assert int(anchors) == 6
    torch.testing.assert_close(summed, by_hand(first[valid], second[valid], TERM.temperature))
    torch.testing.assert_close(scorer.chance(valid), torch.tensor(6 * math.log(5)))


def test_one_pair_or_none_has_no_distractor_and_costs_nothing() -> None:
    scorer = ContrastiveLoss(TERM)
    first, second = torch.randn(3, 8), torch.randn(3, 8)

    for valid in (torch.tensor([True, False, False]), torch.zeros(3, dtype=torch.bool)):
        summed, anchors = scorer.summed(first, second, valid)
        assert float(summed) == 0.0
        assert int(anchors) == 0
        assert float(scorer.chance(valid)) == 0.0


def test_the_contrast_is_read_in_single_precision_whatever_the_projections_came_in() -> None:
    torch.manual_seed(4)
    first, second = torch.randn(4, 8), torch.randn(4, 8)
    scorer = ContrastiveLoss(TERM)
    valid = torch.ones(4, dtype=torch.bool)

    summed, _ = scorer.summed(first.to(torch.bfloat16), second.to(torch.bfloat16), valid)

    assert summed.dtype == torch.float32
    torch.testing.assert_close(
        summed,
        by_hand(first.to(torch.bfloat16).float(), second.to(torch.bfloat16).float(), 0.2),
    )


def test_a_loss_needs_a_stated_term() -> None:
    with pytest.raises(ValueError, match="stated term"):
        ContrastiveLoss(ContrastiveTerm.none())
    with pytest.raises(ValueError, match="needs a term"):
        ContrastiveReconstruction(
            SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE),
            decoder_layers=1,
            contrast=ContrastiveTerm.none(),
        )


def test_the_score_is_the_reconstruction_plus_the_weighed_contrast_over_the_same_tokens(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)
    torch.manual_seed(1)
    plain = MaskedReconstruction(
        SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE), decoder_layers=1
    ).eval()
    reconstruction = plain.score(batch, masks, SQUARED)
    first, second = model.contrasted(batch, masks)
    _, _, valid = model.views(batch, masks)
    contrasting = ContrastiveLoss(model.term)
    contrast_sum, anchors = contrasting.summed(first, second, valid)
    chance = contrasting.chance(valid)
    scale = TERM.weight * float(reconstruction.tokens) / float(anchors)

    score = model.score(batch, masks, SQUARED)

    assert int(score.tokens) == int(reconstruction.tokens)
    torch.testing.assert_close(score.terms[RECONSTRUCTION], reconstruction.error, **TOLERANCE)
    torch.testing.assert_close(score.terms[CONTRAST], contrast_sum * scale, **TOLERANCE)
    torch.testing.assert_close(
        score.error, reconstruction.error + contrast_sum * scale, **TOLERANCE
    )
    torch.testing.assert_close(score.trivial_terms[RECONSTRUCTION], reconstruction.trivial)
    torch.testing.assert_close(score.trivial_terms[CONTRAST], chance * scale, **TOLERANCE)
    torch.testing.assert_close(score.trivial, reconstruction.trivial + chance * scale, **TOLERANCE)
    # Divided by the tokens, the loss is the mean reconstruction plus the weight times the mean
    # contrast per anchor.
    torch.testing.assert_close(
        score.error / score.tokens,
        reconstruction.error / reconstruction.tokens + TERM.weight * contrast_sum / anchors,
        **TOLERANCE,
    )
    assert not score.trivial.requires_grad
    assert all(not term.requires_grad for term in score.terms.values())


def test_a_window_the_masks_hid_nothing_of_is_reconstructed_as_nothing_and_leaves_the_contrast(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(3, 32, seed=7, padding=2)
    masks = masks_over(batch, strategy)
    hidden = masks.hidden.clone()
    hidden[1] = False
    held = masks_hiding(hidden)

    _, _, valid = model.views(batch, held)
    score = model.score(batch, held, SQUARED)

    assert valid.tolist() == [True, False, True]
    assert int(score.tokens) == int((hidden & ~batch.padding_mask).sum())
    torch.testing.assert_close(
        score.trivial_terms[CONTRAST],
        torch.tensor(TERM.weight * math.log(3) * float(score.tokens)),
    )


def test_a_batch_of_one_window_is_scored_by_the_reconstruction_alone(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(1, 32, seed=8, padding=2)
    masks = masks_over(batch, strategy)

    score = model.score(batch, masks, SQUARED)

    assert float(score.terms[CONTRAST]) == 0.0
    assert float(score.trivial_terms[CONTRAST]) == 0.0
    torch.testing.assert_close(score.error, score.terms[RECONSTRUCTION])
    torch.testing.assert_close(score.trivial, score.trivial_terms[RECONSTRUCTION])


def test_every_parameter_of_encoder_decoder_and_projection_receives_a_gradient(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)

    score = model.score(batch, masks, SQUARED)
    (score.error / score.tokens).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_contrast_alone_moves_the_encoder_and_the_projection_and_not_the_decoder(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=9, padding=3)
    masks = masks_over(batch, strategy)
    first, second = model.contrasted(batch, masks)
    _, _, valid = model.views(batch, masks)

    summed, _ = ContrastiveLoss(model.term).summed(first, second, valid)
    summed.backward()

    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name


def test_the_padding_row_of_the_channel_table_receives_none(strategy: MaskingStrategy) -> None:
    model = objective().train()
    batch = random_batch(2, 24, seed=7, padding=4)
    masks = masks_over(batch, strategy)

    model.score(batch, masks, SQUARED).error.backward()

    table = model.encoder.get_parameter("channel_embedding.table.weight").grad
    assert table is not None
    assert not table[0].any()


def test_the_backbone_is_the_encoder_and_nothing_of_the_decoder_or_the_projection() -> None:
    model = objective()

    assert sum(p.numel() for p in model.encoder.parameters()) == SMALL.parameter_count(
        VOCABULARY_SIZE
    )
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {
        "encoder",
        "decoder",
        "projection",
    }
    assert model.term == TERM


def test_the_same_seed_builds_the_same_objective(strategy: MaskingStrategy) -> None:
    batch = random_batch(2, 16, seed=8)
    masks = masks_over(batch, strategy)

    first, second = (
        objective(seed=3).score(batch, masks, SQUARED),
        objective(seed=3).score(batch, masks, SQUARED),
    )

    torch.testing.assert_close(first.error, second.error, rtol=0, atol=0)
    torch.testing.assert_close(first.trivial, second.trivial, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_the_objective_scores_the_same_on_the_accelerator_as_on_the_host(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(4, 32, seed=11, padding=3)
    masks = masks_over(batch, strategy)
    on_host = model.score(batch, masks, SQUARED)

    moved = copy.deepcopy(model).to("mps")
    on_device = moved.score(batch.to("mps"), masks.to("mps"), SQUARED)

    # Off the accelerator first, then widened: the other order reads zeros.
    for host, device in (
        (on_host.error, on_device.error),
        (on_host.trivial, on_device.trivial),
        (on_host.terms[CONTRAST], on_device.terms[CONTRAST]),
    ):
        torch.testing.assert_close(
            device.detach().to("cpu").to(torch.float64),
            host.detach().to(torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )
    assert int(on_device.tokens.to("cpu")) == int(on_host.tokens)
