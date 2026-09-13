import pytest

from emblema.pretraining.domain.exceptions import (
    InvalidMaskingStrategyError,
    PretrainingError,
)
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
