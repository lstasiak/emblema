import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.diagnostics.linear_interpolation_baseline import (  # noqa: E402
    LinearInterpolationBaseline,
)
from tests.pretraining.adapters.diagnostics.conftest import batch_of, hiding, window  # noqa: E402

pytestmark = pytest.mark.ml


def test_a_token_hidden_between_visible_neighbours_of_a_linear_channel_is_recovered_exactly() -> (
    None
):
    batch = batch_of(window({1: lambda t: 2.0 * t - 1.0, 2: lambda t: t * t}, steps=9))
    hidden = (batch.channel_ids == 1) & (batch.timestamps > 0.3) & (batch.timestamps < 0.7)
    masks = hiding(batch, where=hidden)

    prediction = LinearInterpolationBaseline().predict(batch, masks)

    torch.testing.assert_close(prediction[hidden], batch.features[..., 0][hidden])
    assert hidden.sum() == 3


def test_a_curved_channel_is_not_recovered_exactly() -> None:
    batch = batch_of(window({1: lambda t: (2.0 * t - 1.0) ** 2}, steps=5))
    hidden = batch.timestamps == 0.5
    masks = hiding(batch, where=hidden)

    prediction = LinearInterpolationBaseline().predict(batch, masks)

    # Between the neighbours at 0.25 and 0.75 the parabola reads 0.25 while the line reads 0.
    assert prediction[hidden].item() == pytest.approx(0.25)
    assert batch.features[..., 0][hidden].item() == pytest.approx(0.0)


def test_a_channel_hidden_whole_predicts_the_channel_mean() -> None:
    batch = batch_of(window({1: lambda t: 3.0, 2: lambda t: t}, steps=4))
    masks = hiding(batch, channels=[1])

    prediction = LinearInterpolationBaseline().predict(batch, masks)

    assert torch.equal(prediction[masks.hidden], torch.zeros(4))


def test_a_hidden_timeless_token_predicts_the_channel_mean() -> None:
    batch = batch_of(window({1: lambda t: t}, steps=4, timeless={7: 2.5}))
    masks = hiding(batch, channels=[7])

    prediction = LinearInterpolationBaseline().predict(batch, masks)

    assert masks.hidden.sum() == 1
    assert prediction[masks.hidden].item() == 0.0


def test_positions_that_are_not_hidden_predict_nothing_and_padding_is_left_alone() -> None:
    short = window({1: lambda t: t}, steps=3)
    long = window({1: lambda t: t, 2: lambda t: 1.0 - t}, steps=6)
    batch = batch_of(short, long)
    hidden = (batch.channel_ids == 2) & (batch.timestamps == 0.4)
    masks = hiding(batch, where=hidden)

    prediction = LinearInterpolationBaseline().predict(batch, masks)

    assert torch.equal(prediction[~hidden], torch.zeros_like(prediction[~hidden]))
    assert prediction[hidden].item() == pytest.approx(0.6)
    assert prediction.dtype == batch.features.dtype
