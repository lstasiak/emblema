import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.objective.reconstruction_loss import (  # noqa: E402
    ReconstructionLoss,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml


def masks_hiding(hidden: Tensor) -> TokenMasks:
    nothing = torch.zeros_like(hidden)
    return TokenMasks(channel=nothing, block=nothing, token=hidden)


def test_the_loss_is_the_mean_squared_error_over_the_hidden_observed_tokens() -> None:
    batch = random_batch(1, 4, seed=1, padding=1)
    target = batch.features[..., 0]
    prediction = target + torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    hidden = torch.tensor([[True, True, False, True]])

    loss = ReconstructionLoss()(prediction, batch, masks_hiding(hidden))

    # Tokens 0 and 1 are hidden and observed; token 2 was visible; token 3 is padding.
    assert loss.item() == pytest.approx((1.0 + 4.0) / 2)


def test_what_the_prediction_says_at_visible_and_padding_positions_does_not_count() -> None:
    batch = random_batch(3, 20, seed=2, padding=6)
    hidden = torch.rand(3, 20) < 0.5
    prediction = torch.randn(3, 20)
    elsewhere = ~hidden | batch.padding_mask
    perturbed = torch.where(elsewhere, prediction + 100.0, prediction)

    loss = ReconstructionLoss()
    assert torch.equal(
        loss(perturbed, batch, masks_hiding(hidden)), loss(prediction, batch, masks_hiding(hidden))
    )


def test_padding_a_hand_made_mask_claims_as_hidden_is_still_not_scored() -> None:
    batch = random_batch(1, 4, seed=3, padding=2)
    prediction = batch.features[..., 0] + 10.0
    claims_everything = torch.ones(1, 4, dtype=torch.bool)

    loss = ReconstructionLoss()(prediction, batch, masks_hiding(claims_everything))

    assert loss.item() == pytest.approx(100.0)


def test_nothing_hidden_scores_zero_and_backpropagates() -> None:
    batch = random_batch(2, 8, seed=4)
    prediction = torch.randn(2, 8, requires_grad=True)

    loss = ReconstructionLoss()(
        prediction, batch, masks_hiding(torch.zeros(2, 8, dtype=torch.bool))
    )
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_the_mean_runs_over_the_batchs_tokens_not_over_its_windows() -> None:
    batch = random_batch(2, 4, seed=5)
    target = batch.features[..., 0]
    prediction = target + torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0]])
    hidden = torch.tensor([[True, False, False, False], [True, True, True, False]])

    loss = ReconstructionLoss()(prediction, batch, masks_hiding(hidden))

    assert loss.item() == pytest.approx((1.0 + 3 * 9.0) / 4)


def test_over_scores_the_positions_it_is_given() -> None:
    batch = random_batch(1, 3, seed=6)
    prediction = batch.features[..., 0] + torch.tensor([[1.0, 2.0, 3.0]])

    only_last = ReconstructionLoss.over(prediction, batch, torch.tensor([[False, False, True]]))

    assert only_last.item() == pytest.approx(9.0)


def test_the_target_is_read_in_the_predictions_precision() -> None:
    batch = random_batch(1, 3, seed=7)
    prediction = batch.features[..., 0].to(torch.float64)

    loss = ReconstructionLoss()(prediction, batch, masks_hiding(torch.ones(1, 3, dtype=torch.bool)))

    assert loss.dtype == torch.float64
    assert loss.item() == 0.0


def test_the_batch_is_the_source_of_the_target() -> None:
    windows = random_batch(1, 3, seed=8)
    other = TokenTensors(
        windows.features + 1.0,
        windows.channel_ids,
        windows.timestamps,
        windows.timeless,
        windows.padding_mask,
    )
    prediction = windows.features[..., 0]
    everything = masks_hiding(torch.ones(1, 3, dtype=torch.bool))

    assert ReconstructionLoss()(prediction, windows, everything).item() == 0.0
    assert ReconstructionLoss()(prediction, other, everything).item() == pytest.approx(1.0)


def test_the_squared_error_is_per_position_against_the_value_and_masks_nothing() -> None:
    batch = random_batch(2, 5, seed=9, padding=2)
    target = batch.features[..., 0]
    prediction = target + torch.arange(10, dtype=target.dtype).reshape(2, 5)

    squared = ReconstructionLoss.squared_error(prediction, batch)

    torch.testing.assert_close(squared, torch.arange(10, dtype=target.dtype).reshape(2, 5).pow(2))
    everything = torch.ones_like(batch.padding_mask)
    assert ReconstructionLoss.over(prediction, batch, everything).item() == pytest.approx(
        float(squared[~batch.padding_mask].mean())
    )
