import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.objective.reconstruction_loss import (  # noqa: E402
    ReconstructionLoss,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.training.objective_loss import (  # noqa: E402
    LossKind,
    ObjectiveLoss,
)
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml

SQUARED = ObjectiveLoss(kind=LossKind.MSE)
BOUNDED = ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.0)
SCORER = ReconstructionLoss(SQUARED)


def masks_hiding(hidden: Tensor) -> TokenMasks:
    nothing = torch.zeros_like(hidden)
    return TokenMasks(channel=nothing, block=nothing, token=hidden)


def test_the_loss_is_the_mean_squared_error_over_the_hidden_observed_tokens() -> None:
    batch = random_batch(1, 4, seed=1, padding=1)
    target = batch.features[..., 0]
    prediction = target + torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    hidden = torch.tensor([[True, True, False, True]])

    loss = SCORER(prediction, batch, masks_hiding(hidden))

    # Tokens 0 and 1 are hidden and observed; token 2 was visible; token 3 is padding.
    assert loss.item() == pytest.approx((1.0 + 4.0) / 2)


def test_what_the_prediction_says_at_visible_and_padding_positions_does_not_count() -> None:
    batch = random_batch(3, 20, seed=2, padding=6)
    hidden = torch.rand(3, 20) < 0.5
    prediction = torch.randn(3, 20)
    elsewhere = ~hidden | batch.padding_mask
    perturbed = torch.where(elsewhere, prediction + 100.0, prediction)

    assert torch.equal(
        SCORER(perturbed, batch, masks_hiding(hidden)),
        SCORER(prediction, batch, masks_hiding(hidden)),
    )


def test_padding_a_hand_made_mask_claims_as_hidden_is_still_not_scored() -> None:
    batch = random_batch(1, 4, seed=3, padding=2)
    prediction = batch.features[..., 0] + 10.0
    claims_everything = torch.ones(1, 4, dtype=torch.bool)

    loss = SCORER(prediction, batch, masks_hiding(claims_everything))

    assert loss.item() == pytest.approx(100.0)


def test_nothing_hidden_scores_zero_and_backpropagates() -> None:
    batch = random_batch(2, 8, seed=4)
    prediction = torch.randn(2, 8, requires_grad=True)

    loss = SCORER(prediction, batch, masks_hiding(torch.zeros(2, 8, dtype=torch.bool)))
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_the_mean_runs_over_the_batchs_tokens_not_over_its_windows() -> None:
    batch = random_batch(2, 4, seed=5)
    target = batch.features[..., 0]
    prediction = target + torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0]])
    hidden = torch.tensor([[True, False, False, False], [True, True, True, False]])

    loss = SCORER(prediction, batch, masks_hiding(hidden))

    assert loss.item() == pytest.approx((1.0 + 3 * 9.0) / 4)


def test_over_scores_the_positions_it_is_given() -> None:
    batch = random_batch(1, 3, seed=6)
    prediction = batch.features[..., 0] + torch.tensor([[1.0, 2.0, 3.0]])

    only_last = SCORER.over(prediction, batch, torch.tensor([[False, False, True]]))

    assert only_last.item() == pytest.approx(9.0)


def test_the_target_is_read_in_the_predictions_precision() -> None:
    batch = random_batch(1, 3, seed=7)
    prediction = batch.features[..., 0].to(torch.float64)

    loss = SCORER(prediction, batch, masks_hiding(torch.ones(1, 3, dtype=torch.bool)))

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

    assert SCORER(prediction, windows, everything).item() == 0.0
    assert SCORER(prediction, other, everything).item() == pytest.approx(1.0)


def test_the_squared_error_is_per_position_against_the_value_and_masks_nothing() -> None:
    batch = random_batch(2, 5, seed=9, padding=2)
    target = batch.features[..., 0]
    prediction = target + torch.arange(10, dtype=target.dtype).reshape(2, 5)

    squared = SCORER.of_tokens(prediction, batch)

    torch.testing.assert_close(squared, torch.arange(10, dtype=target.dtype).reshape(2, 5).pow(2))
    everything = torch.ones_like(batch.padding_mask)
    assert SCORER.over(prediction, batch, everything).item() == pytest.approx(
        float(squared[~batch.padding_mask].mean())
    )


def test_the_summed_error_is_the_mean_before_it_is_divided() -> None:
    batch = random_batch(3, 12, seed=5, padding=2)
    prediction = torch.randn(3, 12)
    hidden = torch.rand(3, 12) < 0.6

    total, scored = SCORER.summed(prediction, batch, hidden)

    assert total / scored == SCORER.over(prediction, batch, hidden)


def test_a_batch_summed_in_parts_is_the_batch_summed_whole() -> None:
    """What lets a gradient be accumulated: the parts add up to the whole, tokens and all."""
    batch = random_batch(4, 12, seed=6, padding=3)
    prediction = torch.randn(4, 12)
    hidden = torch.rand(4, 12) < 0.6
    whole = SCORER.summed(prediction, batch, hidden)

    parts = [
        SCORER.summed(prediction[rows], _rows(batch, rows), hidden[rows])
        for rows in (slice(0, 2), slice(2, 4))
    ]

    assert sum(total for total, _ in parts) == pytest.approx(whole[0].item())
    assert sum(int(scored) for _, scored in parts) == int(whole[1])


def _rows(batch: TokenTensors, rows: slice) -> TokenTensors:
    return TokenTensors(
        features=batch.features[rows],
        channel_ids=batch.channel_ids[rows],
        timestamps=batch.timestamps[rows],
        timeless=batch.timeless[rows],
        padding_mask=batch.padding_mask[rows],
    )


@pytest.mark.parametrize("reading", [SQUARED, BOUNDED])
def test_the_tensors_agree_with_the_definition_the_domain_states(
    reading: ObjectiveLoss,
) -> None:
    """One token at a time, the arithmetic over a batch is what ``of_error`` says it is."""
    batch = random_batch(2, 6, seed=11)
    target = batch.features[..., 0]
    prediction = target + torch.linspace(-20.0, 20.0, 12).reshape(2, 6)

    of_tokens = ReconstructionLoss(reading).of_tokens(prediction, batch)

    expected = [
        [reading.of_error(float(prediction[row, token] - target[row, token])) for token in range(6)]
        for row in range(2)
    ]
    torch.testing.assert_close(of_tokens, torch.tensor(expected, dtype=of_tokens.dtype))


def test_the_bounded_reading_stops_one_token_from_deciding_the_gradient() -> None:
    """What the bounded reading is for: an excursion pulls no harder than the knee lets it."""
    batch = random_batch(1, 3, seed=12)
    hidden = torch.ones(1, 3, dtype=torch.bool)
    ordinary = batch.features[..., 0] + torch.tensor([[0.5, 0.5, 0.5]])
    excursion = batch.features[..., 0] + torch.tensor([[0.5, 0.5, 100.0]])

    def pull(scorer: ReconstructionLoss, prediction: Tensor) -> float:
        asked = prediction.detach().requires_grad_()
        scorer(asked, batch, masks_hiding(hidden)).backward()
        assert asked.grad is not None
        return float(asked.grad[0, 2].abs())

    assert pull(SCORER, excursion) / pull(SCORER, ordinary) == pytest.approx(200.0)
    bounded = ReconstructionLoss(BOUNDED)
    assert pull(bounded, excursion) == pytest.approx(pull(bounded, ordinary) * 2.0)


def test_the_trivial_predictor_is_read_under_the_same_reading_as_the_model() -> None:
    batch = random_batch(2, 5, seed=13, padding=1)
    target = batch.features[..., 0]
    hidden = torch.ones_like(batch.padding_mask)

    for scorer, reading in ((SCORER, SQUARED), (ReconstructionLoss(BOUNDED), BOUNDED)):
        nothing_learnt, scored = scorer.summed_over_mean(batch, hidden)

        # The rule the model is scored by, applied to the predictor that knows nothing: padding
        # is not a token, and the reading is the run's.
        observed = target[~batch.padding_mask]
        assert int(scored) == int(observed.numel())
        expected = sum(reading.of_error(float(value)) for value in observed)
        assert float(nothing_learnt) == pytest.approx(expected, rel=1e-6)
