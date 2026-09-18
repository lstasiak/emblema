import pytest

from emblema.shared.kernel.tokens import Token, TokenWindow

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.pretraining.adapters.encoder.fourier_time_encoding import (  # noqa: E402
    FourierTimeEncoding,
)
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (  # noqa: E402
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import (  # noqa: E402
    VOCABULARY_SIZE,
    all_timeless,
    fully_padded,
    padded_by,
    permutation,
    permuted,
    random_batch,
    scrambled_under_padding,
    with_timestamps,
)

pytestmark = pytest.mark.ml

# Reordering or padding a batch changes the order attention reduces in, so two runs over the same
# tokens agree in float32 to the precision of a reduction, not bit for bit.
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def test_one_batch_holds_windows_of_different_channel_counts_and_token_counts(
    encoder: SetEncoder,
) -> None:
    few_channels = TokenWindow.of(
        Token(channel_id=channel, value=0.1 * channel, time=step / 4, gap=0.25 if step else 0.0)
        for channel in (1, 2, 3)
        for step in range(4)
    )
    many_channels = TokenWindow.of(
        [Token(channel_id=channel, value=0.0, time=0.5, gap=0.5) for channel in range(1, 31)]
        + [Token(channel_id=40, value=1.0, time=0.0, gap=0.0, timeless=True)]
    )
    batch = TokenTensors.from_windows([few_channels, many_channels])

    states = encoder(*batch.args)

    assert states.shape == (2, len(many_channels), SMALL.width)
    assert torch.isfinite(states).all()
    # Sharing a batch with a longer window must cost the shorter one nothing: the padding that
    # squares them off is the only difference between the two calls.
    alone = encoder(*TokenTensors.from_windows([few_channels]).args)
    torch.testing.assert_close(states[0, : len(few_channels)], alone[0], **TOLERANCE)


def test_permuting_the_tokens_permutes_their_states(encoder: SetEncoder) -> None:
    batch = random_batch(2, 64, seed=11)
    order = permutation(batch, seed=3)

    states = encoder(*batch.args)
    states_of_permuted = encoder(*permuted(batch, seed=3).args)

    torch.testing.assert_close(states_of_permuted, states[:, order], **TOLERANCE)


def test_the_pooled_embedding_ignores_the_order_of_the_tokens(encoder: SetEncoder) -> None:
    pooling = MaskedMeanPooling()
    batch = random_batch(2, 64, seed=11)
    shuffled = permuted(batch, seed=3)

    torch.testing.assert_close(
        pooling(encoder(*shuffled.args), shuffled.padding_mask),
        pooling(encoder(*batch.args), batch.padding_mask),
        **TOLERANCE,
    )


def test_padding_appended_to_a_window_leaves_the_observed_states_unchanged(
    encoder: SetEncoder,
) -> None:
    batch = random_batch(2, 40, seed=13)

    states = encoder(*batch.args)
    extended = encoder(*padded_by(batch, 24).args)

    torch.testing.assert_close(extended[:, :40], states, **TOLERANCE)


def test_what_a_padding_position_carries_does_not_reach_the_observed_tokens(
    encoder: SetEncoder,
) -> None:
    batch = random_batch(3, 41, seed=41, padding=17)
    observed = ~batch.padding_mask

    states = encoder(*batch.args)
    scrambled = encoder(*scrambled_under_padding(batch, seed=5).args)

    torch.testing.assert_close(scrambled[observed], states[observed], **TOLERANCE)


def test_a_timeless_token_ignores_its_timestamp(encoder: SetEncoder) -> None:
    batch = all_timeless(random_batch(1, 32, seed=5))

    torch.testing.assert_close(
        encoder(*with_timestamps(batch, seed=6).args), encoder(*batch.args), rtol=0, atol=0
    )


def test_a_window_of_nothing_but_padding_yields_finite_states(encoder: SetEncoder) -> None:
    # Attention over an entirely masked window is a softmax over nothing; the mechanism was chosen
    # for returning numbers here rather than NaN, and this holds it to that.
    states = encoder(*fully_padded(random_batch(1, 16, seed=7)).args)

    assert torch.isfinite(states).all()


def test_every_parameter_receives_a_gradient() -> None:
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE)
    batch = random_batch(2, 40, seed=13, padding=5)

    MaskedMeanPooling()(encoder(*batch.args), batch.padding_mask).pow(2).mean().backward()

    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_model_holds_exactly_the_parameters_the_architecture_counts(
    encoder: SetEncoder,
) -> None:
    held = sum(parameter.numel() for parameter in encoder.parameters())

    assert held == SMALL.parameter_count(VOCABULARY_SIZE)


def test_the_same_seed_builds_the_same_encoder() -> None:
    batch = random_batch(1, 16, seed=2)
    outputs = []
    for _ in range(2):
        torch.manual_seed(7)
        outputs.append(SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE).eval()(*batch.args))

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_dropout_acts_in_training_only() -> None:
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE, dropout=0.5)
    batch = random_batch(1, 32, seed=3)

    encoder.train()
    assert not torch.equal(encoder(*batch.args), encoder(*batch.args))
    encoder.eval()
    assert torch.equal(encoder(*batch.args), encoder(*batch.args))


class _NoTimeEncoding(nn.Module):
    def forward(self, timestamps: Tensor, timeless: Tensor) -> Tensor:
        return timestamps.new_zeros((*timestamps.shape, SMALL.width))


def test_a_variant_input_module_takes_the_place_of_the_standard_one() -> None:
    # The boundary a learned time encoding or a described channel embedding would enter through:
    # with a module that encodes no time at all, the encoder stops seeing timestamps entirely.
    torch.manual_seed(1)
    encoder = SetEncoder(
        SMALL,
        channel_embedding=LearnedChannelEmbedding(VOCABULARY_SIZE, SMALL.width),
        time_encoding=_NoTimeEncoding(),
    ).eval()
    batch = random_batch(1, 16, seed=4)

    torch.testing.assert_close(
        encoder(*with_timestamps(batch, seed=9).args), encoder(*batch.args), rtol=0, atol=0
    )
    assert not isinstance(encoder.time_encoding, FourierTimeEncoding)
