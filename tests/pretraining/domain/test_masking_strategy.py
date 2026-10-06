import pytest

from emblema.pretraining.domain.exceptions import (
    InvalidMaskingStrategyError,
    PretrainingError,
)
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy


def strategy(**overrides: float) -> MaskingStrategy:
    dials: dict[str, float] = {
        "channel_rate": 0.15,
        "block_rate": 0.6,
        "block_span": 0.5,
        "token_rate": 0.1,
    }
    return MaskingStrategy(**{**dials, **overrides})


def test_the_expected_ratio_is_what_survives_no_draw() -> None:
    expected = 1.0 - (1.0 - 0.15) * (1.0 - 0.6 * 0.5) * (1.0 - 0.1)

    assert strategy().expected_ratio == pytest.approx(expected)


def test_a_single_draw_reports_its_own_rate_as_the_ratio() -> None:
    assert strategy(channel_rate=0.0, block_rate=0.0, token_rate=0.3).expected_ratio == (
        pytest.approx(0.3)
    )
    assert strategy(channel_rate=0.0, block_rate=0.5, token_rate=0.0).expected_ratio == (
        pytest.approx(0.25)
    )


@pytest.mark.parametrize("label", ["channel_rate", "block_rate", "token_rate"])
@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_a_rate_outside_the_unit_interval_is_rejected(label: str, rate: float) -> None:
    with pytest.raises(InvalidMaskingStrategyError, match=label):
        strategy(**{label: rate})


@pytest.mark.parametrize("span", [0.0, -0.5, 1.1])
def test_a_block_span_outside_the_window_is_rejected(span: float) -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="block_span"):
        strategy(block_span=span)


def test_a_strategy_that_hides_nothing_is_rejected() -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="hide something"):
        strategy(channel_rate=0.0, block_rate=0.0, token_rate=0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel_rate": 1.0},
        {"token_rate": 1.0},
        {"block_rate": 1.0, "block_span": 1.0},
    ],
)
def test_a_strategy_that_hides_everything_for_certain_is_rejected(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="leave something visible"):
        strategy(**overrides)


def test_the_error_is_a_value_error_of_the_context() -> None:
    with pytest.raises(PretrainingError, match="token_rate"):
        strategy(token_rate=2.0)
    with pytest.raises(ValueError, match="token_rate"):
        strategy(token_rate=2.0)


def tail(rate: float = 1.0, shortest: float = 0.15, longest: float = 0.5) -> MaskingStrategy:
    return strategy(
        block_rate=0.0,
        token_rate=0.0,
        horizon_rate=rate,
        horizon_min_span=shortest,
        horizon_max_span=longest,
    )


def test_the_tail_hides_its_mean_span_at_its_rate() -> None:
    assert tail().expected_ratio == pytest.approx(1.0 - 0.85 * (1.0 - 0.325))
    assert tail(rate=0.5).expected_ratio == pytest.approx(1.0 - 0.85 * (1.0 - 0.5 * 0.325))


def test_a_strategy_without_a_tail_hides_what_it_hid_before() -> None:
    assert not strategy().has_horizon
    assert strategy().expected_ratio == pytest.approx(0.4645)


@pytest.mark.parametrize(("shortest", "longest"), [(0.0, 0.5), (0.4, 0.3), (0.2, 1.0), (-0.1, 0.5)])
def test_tail_spans_outside_the_window_or_out_of_order_are_rejected(
    shortest: float, longest: float
) -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="horizon spans"):
        tail(shortest=shortest, longest=longest)


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_a_tail_rate_outside_the_unit_interval_is_rejected(rate: float) -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="horizon_rate"):
        tail(rate=rate)


def test_a_tail_that_is_never_drawn_states_no_spans() -> None:
    with pytest.raises(InvalidMaskingStrategyError, match="never drawn"):
        strategy(horizon_min_span=0.1, horizon_max_span=0.2)


def test_a_tail_alone_is_a_strategy_and_one_of_fixed_length_is_too() -> None:
    alone = MaskingStrategy(
        channel_rate=0.0,
        block_rate=0.0,
        block_span=0.5,
        token_rate=0.0,
        horizon_rate=1.0,
        horizon_min_span=0.3,
        horizon_max_span=0.3,
    )

    assert alone.expected_ratio == pytest.approx(0.3)


def test_a_strategy_draws_the_kinds_whose_rates_are_positive() -> None:
    assert [kind for kind in MaskKind if strategy().draws(kind)] == [
        MaskKind.CHANNEL,
        MaskKind.BLOCK,
        MaskKind.TOKEN,
    ]
    assert [kind for kind in MaskKind if tail().draws(kind)] == [
        MaskKind.CHANNEL,
        MaskKind.HORIZON,
    ]
