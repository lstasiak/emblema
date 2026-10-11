import copy

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402
from torch.nn import functional  # noqa: E402

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.adapters.objective.latent_prediction_loss import (  # noqa: E402
    LatentPredictionLoss,
)
from emblema.pretraining.adapters.objective.latent_predictor import LatentPredictor  # noqa: E402
from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def objective(
    *, seed: int = 1, layers: int = 2, momentum: float = 0.9, predictor_layers: int = 1
) -> LatentPrediction:
    torch.manual_seed(seed)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)
    target = ObjectiveTarget.latent(layers=layers, momentum=momentum, final_momentum=1.0)
    return LatentPrediction(
        encoder, predictor_layers=predictor_layers, target=target, dropout=0.1
    ).eval()


def masks_over(batch: TokenTensors, strategy: MaskingStrategy, *, seed: int = 1) -> TokenMasks:
    return TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(seed))


def with_features(batch: TokenTensors, features: Tensor) -> TokenTensors:
    return TokenTensors(
        features, batch.channel_ids, batch.timestamps, batch.timeless, batch.padding_mask
    )


def test_the_prediction_and_the_target_are_one_state_per_position(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(3, 40, seed=2, padding=7)

    prediction = model(batch, masks_over(batch, strategy))
    target = model.targets(batch)

    assert prediction.shape == target.shape == (3, 40, SMALL.width)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(target).all()
    assert not target.requires_grad


def test_the_encoder_never_sees_the_value_of_a_hidden_token(strategy: MaskingStrategy) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=3, padding=5)
    masks = masks_over(batch, strategy)
    other = random_batch(2, 48, seed=4)
    scrambled = with_features(
        batch, torch.where(masks.hidden.unsqueeze(-1), other.features, batch.features)
    )

    torch.testing.assert_close(model(scrambled, masks), model(batch, masks), **TOLERANCE)


def test_the_teacher_sees_the_whole_window_hidden_tokens_included(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=3, padding=5)
    masks = masks_over(batch, strategy)
    other = random_batch(2, 48, seed=4)
    scrambled = with_features(
        batch, torch.where(masks.hidden.unsqueeze(-1), other.features, batch.features)
    )

    assert not torch.allclose(model.targets(scrambled), model.targets(batch), **TOLERANCE)


def test_the_target_is_the_mean_of_the_top_blocks_normalised_without_the_learnt_scale(
    strategy: MaskingStrategy,
) -> None:
    model = objective(layers=1)
    batch = random_batch(2, 24, seed=5)

    stream = model.teacher.residual_states(*batch.args)
    expected = functional.layer_norm(stream[-1], (SMALL.width,))
    torch.testing.assert_close(model.targets(batch), expected)

    both = objective(layers=2)
    averaged = functional.layer_norm(stream[-2:], (SMALL.width,)).mean(0)
    torch.testing.assert_close(both.targets(batch), averaged)
    # The embedding layer is never a target, however many blocks are read.
    assert not torch.allclose(both.targets(batch), functional.layer_norm(stream[0], (SMALL.width,)))


def test_the_prediction_for_a_hidden_token_depends_on_which_channel_and_when(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(2, 48, seed=5)
    masks = masks_over(batch, strategy)
    other_channels = TokenTensors(
        batch.features,
        torch.where(masks.hidden, (batch.channel_ids % VOCABULARY_SIZE) + 1, batch.channel_ids),
        batch.timestamps,
        batch.timeless,
        batch.padding_mask,
    )

    prediction = model(batch, masks)
    relabelled = model(other_channels, masks)

    assert not torch.allclose(prediction[masks.hidden], relabelled[masks.hidden], **TOLERANCE)


def test_every_parameter_of_encoder_and_predictor_receives_a_gradient_and_none_of_the_teacher(
    strategy: MaskingStrategy,
) -> None:
    model = objective().train()
    batch = random_batch(4, 40, seed=6, padding=3)
    masks = masks_over(batch, strategy)

    score = model.score(batch, masks, SQUARED)
    (score.error / score.tokens).backward()

    for name, parameter in model.named_parameters():
        if name.startswith("teacher."):
            assert parameter.grad is None, name
            assert not parameter.requires_grad, name
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_padding_row_of_the_channel_table_receives_none(strategy: MaskingStrategy) -> None:
    model = objective().train()
    batch = random_batch(2, 24, seed=7, padding=4)
    masks = masks_over(batch, strategy)

    model.score(batch, masks, SQUARED).error.backward()

    table = model.encoder.get_parameter("channel_embedding.table.weight").grad
    assert table is not None
    assert not table[0].any()


def test_the_loss_is_scored_on_hidden_observed_tokens_and_on_nothing_else() -> None:
    model = objective()
    batch = random_batch(2, 12, seed=8, padding=4)
    hidden = torch.zeros(2, 12, dtype=torch.bool)
    hidden[0, :3] = True  # observed and hidden
    hidden[1, -2:] = True  # hidden under the padding, which is no token
    masks = TokenMasks(
        channel=torch.zeros_like(hidden),
        block=hidden,
        token=torch.zeros_like(hidden),
        horizon=torch.zeros_like(hidden),
    )
    scorer = LatentPredictionLoss(SQUARED)
    prediction, target = model(batch, masks), model.targets(batch)

    error, tokens = scorer.summed(prediction, target, batch, masks.hidden)

    assert int(tokens) == 3
    by_hand = ((prediction[0, :3] - target[0, :3]) ** 2).mean(-1).sum()
    torch.testing.assert_close(error, by_hand)
    # The visible tokens' distance does not enter, however large.
    far = prediction.clone()
    far[0, 3:] += 100.0
    torch.testing.assert_close(scorer.summed(far, target, batch, masks.hidden)[0], error)


def test_the_trivial_predictor_reads_no_hidden_token_s_target(strategy: MaskingStrategy) -> None:
    model = objective()
    batch = random_batch(2, 32, seed=9, padding=3)
    masks = masks_over(batch, strategy)
    scorer = LatentPredictionLoss(SQUARED)
    target = model.targets(batch)
    visible = (~batch.padding_mask & ~masks.hidden).unsqueeze(-1)
    moved = torch.where(visible, target, target + 5.0)

    trivial, counted = scorer.summed_over_visible_mean(target, batch, masks, masks.hidden)
    under_moved, _ = scorer.summed_over_visible_mean(moved, batch, masks, masks.hidden)

    assert int(counted) == int((masks.hidden & ~batch.padding_mask).sum())
    mean = (target * visible).sum(1, keepdim=True) / visible.sum(1, keepdim=True)
    by_hand, _ = scorer.summed(mean.expand_as(target), target, batch, masks.hidden)
    torch.testing.assert_close(trivial, by_hand)
    # Moving every hidden token's target by the same vector moves the trivial loss by its
    # square, and the mean it is read against not at all.
    torch.testing.assert_close(
        under_moved - trivial, torch.tensor(25.0) * counted.to(trivial.dtype)
    )


def test_the_teacher_moves_toward_the_encoder_by_the_momentum_after_a_step() -> None:
    model = objective(momentum=0.9)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1.0)
    before = [p.clone() for p in model.teacher.parameters()]

    model.after_step(0, 100)

    for trailing, was, trained in zip(
        model.teacher.parameters(), before, model.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(trailing, 0.9 * was + 0.1 * trained)


def test_a_momentum_of_one_leaves_the_teacher_where_it_was() -> None:
    model = objective(momentum=0.9)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1.0)
    before = [p.clone() for p in model.teacher.parameters()]

    model.after_step(100, 100)

    for trailing, was in zip(model.teacher.parameters(), before, strict=True):
        torch.testing.assert_close(trailing, was, rtol=0, atol=0)


def test_the_teacher_stays_in_evaluation_mode_while_the_rest_trains() -> None:
    model = objective().train()

    assert model.training
    assert model.encoder.training
    assert model.predictor.training
    assert not model.teacher.training
    assert all(not module.training for module in model.teacher.modules())

    batch = random_batch(2, 24, seed=10)
    torch.testing.assert_close(model.targets(batch), model.targets(batch), rtol=0, atol=0)


def test_the_backbone_is_the_encoder_and_nothing_of_the_predictor_or_the_teacher() -> None:
    model = objective(predictor_layers=2)

    assert sum(p.numel() for p in model.encoder.parameters()) == SMALL.parameter_count(
        VOCABULARY_SIZE
    )
    assert sum(p.numel() for p in model.teacher.parameters()) == SMALL.parameter_count(
        VOCABULARY_SIZE
    )
    assert not any(name.startswith("encoder.") for name, _ in model.predictor.named_parameters())
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == {
        "encoder",
        "predictor",
        "teacher",
    }


def test_a_target_the_encoder_cannot_give_is_refused() -> None:
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)

    with pytest.raises(ValueError, match="blocks of an encoder"):
        LatentPrediction(
            encoder, predictor_layers=1, target=ObjectiveTarget.latent(layers=SMALL.layers + 1)
        )
    with pytest.raises(ValueError, match="predicts states"):
        LatentPrediction(encoder, predictor_layers=1, target=ObjectiveTarget.value())
    with pytest.raises(ValueError, match="at least one block"):
        LatentPredictor(SMALL, layers=0)


def test_the_same_seed_builds_the_same_objective(strategy: MaskingStrategy) -> None:
    batch = random_batch(1, 16, seed=8)
    masks = masks_over(batch, strategy)

    torch.testing.assert_close(
        objective(seed=3)(batch, masks), objective(seed=3)(batch, masks), rtol=0, atol=0
    )
    torch.testing.assert_close(
        objective(seed=3).targets(batch), objective(seed=3).targets(batch), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_the_objective_scores_the_same_on_the_accelerator_as_on_the_host(
    strategy: MaskingStrategy,
) -> None:
    model = objective()
    batch = random_batch(2, 32, seed=11, padding=3)
    masks = masks_over(batch, strategy)
    on_host = model.score(batch, masks, SQUARED)

    moved = copy.deepcopy(model).to("mps")
    on_device = moved.score(batch.to("mps"), masks.to("mps"), SQUARED)
    moved.after_step(0, 10)

    # Off the accelerator first, then widened: the other order reads zeros.
    for host, device in ((on_host.error, on_device.error), (on_host.trivial, on_device.trivial)):
        torch.testing.assert_close(
            device.detach().to("cpu").to(torch.float64),
            host.detach().to(torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )
    assert int(on_device.tokens.to("cpu")) == int(on_host.tokens)
    for trailing, trained in zip(
        moved.teacher.parameters(), moved.encoder.parameters(), strict=True
    ):
        assert trailing.device.type == trained.device.type == "mps"
