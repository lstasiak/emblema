import math

import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.diagnostics.cross_channel_ridge_baseline import (  # noqa: E402
    CrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.shared.kernel.tokens import TokenWindow  # noqa: E402
from tests.pretraining.adapters.diagnostics.conftest import batch_of, hiding, window  # noqa: E402

pytestmark = pytest.mark.ml

VOCABULARY = 4


def coupled(phase: float, *, steps: int = 12) -> TokenWindow:
    # Channel 3 is exactly twice channel 1 minus channel 2, plus a half.
    first = lambda t: math.sin(2 * math.pi * t + phase)  # noqa: E731
    second = lambda t: math.cos(3 * math.pi * t + phase)  # noqa: E731
    return window({1: first, 2: second, 3: lambda t: 2 * first(t) - second(t) + 0.5}, steps=steps)


def test_a_channel_that_is_a_linear_combination_of_the_others_is_recovered() -> None:
    training = [batch_of(*(coupled(phase / 10) for phase in range(10)))]
    held_out = batch_of(coupled(1.234))
    masks = hiding(held_out, channels=[3])

    baseline = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY, penalty=1e-9)
    prediction = baseline.predict(held_out, masks)

    torch.testing.assert_close(
        prediction[masks.hidden], held_out.features[..., 0][masks.hidden], atol=1e-4, rtol=1e-4
    )
    weights = baseline.coefficients[3]
    assert weights[[1, 2, -1]].tolist() == pytest.approx([2.0, -1.0, 0.5], abs=1e-4)


def test_a_channel_never_explains_itself() -> None:
    training = [batch_of(*(coupled(phase / 10) for phase in range(10)))]

    baseline = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY)

    for channel in (1, 2, 3):
        assert baseline.coefficients[channel, channel] == 0.0


def test_a_channel_the_fit_never_saw_predicts_the_channel_mean() -> None:
    training = [batch_of(coupled(0.0))]
    held_out = batch_of(window({1: lambda t: t, 4: lambda t: 5.0}, steps=6))
    masks = hiding(held_out, channels=[4])

    baseline = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY)

    assert torch.equal(baseline.predict(held_out, masks)[masks.hidden], torch.zeros(6))
    assert not baseline.coefficients[4].any()


def test_the_features_are_the_other_channels_values_at_their_nearest_visible_instants() -> None:
    # Channel 2 mirrors channel 1 but is sampled at other instants; at prediction time only
    # channel 1 is visible and the fit has to read it where it was actually observed.
    training = [
        batch_of(
            window({1: lambda t: t, 2: lambda t: t}, steps=11),
            window({1: lambda t: 1 - t, 2: lambda t: 1 - t}, steps=11),
        )
    ]
    dense = window({1: lambda t: t}, steps=6, instants=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sparse = window({2: lambda t: t}, steps=3, instants=[0.1, 0.5, 0.9])
    held_out = batch_of(TokenWindow.of([*dense, *sparse]))
    masks = hiding(held_out, channels=[2])

    baseline = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY, penalty=1e-9)
    prediction = baseline.predict(held_out, masks)

    # Nearest visible instants of channel 1 to 0.1, 0.5 and 0.9 are 0.0 or 0.2, 0.4 or 0.6, and
    # 0.8 or 1.0 — a tie broken towards the earlier one.
    torch.testing.assert_close(
        prediction[masks.hidden], torch.tensor([0.0, 0.4, 0.8]), atol=1e-4, rtol=1e-4
    )


def test_penalty_and_vocabulary_are_validated() -> None:
    with pytest.raises(ValueError, match="vocabulary size"):
        CrossChannelRidgeBaseline.fitted([], vocabulary_size=0)
    with pytest.raises(ValueError, match="penalty"):
        CrossChannelRidgeBaseline.fitted([], vocabulary_size=1, penalty=-1.0)


def test_the_penalty_shrinks_the_weights_but_not_the_bias() -> None:
    training = [batch_of(*(coupled(phase / 10) for phase in range(10)))]
    values = training[0].features[..., 0]
    mean_of_target = values[training[0].channel_ids == 3].mean().item()

    loose = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY, penalty=1e-9)
    tight = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY, penalty=1e9)

    assert abs(tight.coefficients[3, 1]) < 1e-6 < abs(loose.coefficients[3, 1])
    # With the weights crushed, the free bias is left to explain the target on its own.
    assert tight.coefficients[3, -1] == pytest.approx(mean_of_target, abs=1e-4)


def test_a_window_with_nothing_hidden_predicts_nothing() -> None:
    training = [batch_of(coupled(0.0))]
    held_out = batch_of(coupled(0.5), coupled(1.5))
    only_second = hiding(held_out, channels=[3])
    only_second = TokenMasks(
        channel=only_second.channel & (torch.arange(2)[:, None] == 1),
        block=only_second.block,
        token=only_second.token,
    )

    baseline = CrossChannelRidgeBaseline.fitted(training, vocabulary_size=VOCABULARY)
    prediction = baseline.predict(held_out, only_second)

    assert not prediction[0].any()
    assert prediction[1][only_second.hidden[1]].abs().sum() > 0
