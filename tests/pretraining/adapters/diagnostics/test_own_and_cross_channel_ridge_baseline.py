import math

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.diagnostics.own_and_cross_channel_ridge_baseline import (  # noqa: E402
    OwnAndCrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.mask_kind import MaskKind  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.tokens import TokenWindow  # noqa: E402
from tests.pretraining.adapters.diagnostics.conftest import batch_of, window  # noqa: E402

pytestmark = pytest.mark.ml

VOCABULARY = 4
STEPS = 13
# Column of the channel's own interpolated value in a row of coefficients: after one weight per
# vocabulary entry, the padding entry included.
OWN_LINE = VOCABULARY + 1


def masks(
    batch: TokenTensors, *, block: Tensor | None = None, token: Tensor | None = None
) -> TokenMasks:
    nothing = torch.zeros_like(batch.padding_mask)
    return TokenMasks(
        channel=nothing,
        block=(block if block is not None else nothing) & ~batch.padding_mask,
        token=(token if token is not None else nothing) & ~batch.padding_mask,
    )


def middle(batch: TokenTensors, channel: int) -> Tensor:
    return (batch.channel_ids == channel) & (batch.timestamps > 0.3) & (batch.timestamps < 0.7)


def lines(count: int, *, offset: float = 0.0) -> list[TokenWindow]:
    """Channel 1 a straight line of its own slope per window, channel 2 unrelated to it."""

    def of(index: int) -> TokenWindow:
        slope = math.sin(index + offset) * 2.0
        return window(
            {1: lambda t: slope * t - slope / 2, 2: lambda t: math.cos(7 * t + index + offset)},
            steps=STEPS,
        )

    return [of(index) for index in range(count)]


def test_a_single_token_on_a_straight_channel_is_read_off_its_own_line() -> None:
    training = batch_of(*lines(20))
    held_out = batch_of(*lines(3, offset=0.5))
    quarter_token = lambda batch: (batch.channel_ids == 1) & (batch.timestamps == 0.25)  # noqa: E731

    baseline = OwnAndCrossChannelRidgeBaseline.fitted(
        [(training, masks(training, token=quarter_token(training)))],
        vocabulary_size=VOCABULARY,
        penalty=1e-9,
    )
    hidden = masks(held_out, token=quarter_token(held_out))
    prediction = baseline.predict(held_out, hidden)

    torch.testing.assert_close(
        prediction[hidden.hidden], held_out.features[..., 0][hidden.hidden], atol=1e-4, rtol=1e-4
    )
    assert baseline.coefficients[MaskKind.TOKEN][1, OWN_LINE] == pytest.approx(1.0, abs=1e-3)


def echo(phase: float) -> TokenWindow:
    """Channel 3 repeats channel 1, which curves too much for a line across half the window."""
    signal = lambda t: math.sin(6 * t + phase)  # noqa: E731
    return window({1: signal, 3: signal}, steps=STEPS)


def test_a_block_is_read_off_the_channel_that_repeats_it_rather_than_its_own_line() -> None:
    training = batch_of(*(echo(phase / 3) for phase in range(20)))
    held_out = batch_of(echo(0.123), echo(2.5))

    baseline = OwnAndCrossChannelRidgeBaseline.fitted(
        [(training, masks(training, block=middle(training, 3)))],
        vocabulary_size=VOCABULARY,
        penalty=1e-9,
    )
    hidden = masks(held_out, block=middle(held_out, 3))
    prediction = baseline.predict(held_out, hidden)

    torch.testing.assert_close(
        prediction[hidden.hidden], held_out.features[..., 0][hidden.hidden], atol=1e-4, rtol=1e-4
    )
    weights = baseline.coefficients[MaskKind.BLOCK][3]
    assert weights[1] == pytest.approx(1.0, abs=1e-3)
    assert weights[OWN_LINE] == pytest.approx(0.0, abs=1e-3)


def test_the_fit_learns_from_hidden_tokens_only_so_a_token_never_interpolates_itself() -> None:
    # Channel 1 is noise with no structure in time: the line between visible neighbours says
    # nothing about a hidden token. Were visible tokens targets too, each would be interpolated
    # from itself and the weight on the line would be pulled towards one.
    generator = torch.Generator().manual_seed(3)

    def noise() -> TokenWindow:
        values = torch.randn(STEPS, generator=generator).tolist()

        def at(time: float) -> float:
            return float(values[round(time * (STEPS - 1))])

        return window({1: at}, steps=STEPS)

    training = batch_of(*(noise() for _ in range(40)))

    baseline = OwnAndCrossChannelRidgeBaseline.fitted(
        [(training, masks(training, block=middle(training, 1)))], vocabulary_size=VOCABULARY
    )

    assert abs(baseline.coefficients[MaskKind.BLOCK][1, OWN_LINE]) < 0.3


def test_a_token_of_a_channel_hidden_whole_is_left_to_the_cross_channel_baseline() -> None:
    training = batch_of(*(echo(phase / 3) for phase in range(5)))
    held_out = batch_of(echo(1.0))
    whole = held_out.channel_ids == 3
    nothing = torch.zeros_like(whole)

    baseline = OwnAndCrossChannelRidgeBaseline.fitted(
        [(training, masks(training, block=middle(training, 3)))], vocabulary_size=VOCABULARY
    )
    prediction = baseline.predict(held_out, TokenMasks(channel=whole, block=nothing, token=nothing))

    assert not prediction.any()


def test_penalty_and_vocabulary_are_validated() -> None:
    with pytest.raises(ValueError, match="vocabulary size"):
        OwnAndCrossChannelRidgeBaseline.fitted([], vocabulary_size=0)
    with pytest.raises(ValueError, match="penalty"):
        OwnAndCrossChannelRidgeBaseline.fitted([], vocabulary_size=1, penalty=-1.0)
