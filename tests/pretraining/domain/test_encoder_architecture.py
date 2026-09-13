from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import InvalidEncoderArchitectureError

SMALL = EncoderArchitecture(width=32, heads=4, layers=2, feedforward_width=64, time_frequencies=8)
# The reference shape of the budget arithmetic: 256 wide, 6 blocks, feed-forward four times the
# width. Its dense parameters come to 12 · 256² · 6 = 4 718 592 there; the exact count below adds
# the channel table, the input projections, the biases and the normalisations. The vocabulary it is
# counted over is an arbitrary one — how many channels a corpus mix has is not a fact of the domain.
REFERENCE = EncoderArchitecture(
    width=256, heads=4, layers=6, feedforward_width=1024, time_frequencies=12
)

architectures = st.builds(
    lambda heads, head_width, layers, hidden, frequencies: EncoderArchitecture(
        width=heads * head_width,
        heads=heads,
        layers=layers,
        feedforward_width=hidden,
        time_frequencies=frequencies,
    ),
    heads=st.integers(1, 8),
    head_width=st.integers(1, 64),
    layers=st.integers(1, 12),
    hidden=st.integers(1, 2048),
    frequencies=st.integers(1, 32),
)


def test_the_reference_shape_counts_its_parameters_exactly() -> None:
    assert REFERENCE.parameter_count(64) == 4_762_880


# In a process that has already loaded torch the first draw takes over a second — the library's
# one-off setup, not this strategy — and the health check cannot tell the two apart.
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(architecture=architectures, vocabulary_size=st.integers(1, 10_000))
def test_one_more_channel_costs_one_more_row_of_the_width(
    architecture: EncoderArchitecture, vocabulary_size: int
) -> None:
    grown = architecture.parameter_count(vocabulary_size + 1)

    assert grown - architecture.parameter_count(vocabulary_size) == architecture.width


@pytest.mark.parametrize(
    "field", ["width", "heads", "layers", "feedforward_width", "time_frequencies"]
)
@pytest.mark.parametrize("count", [0, -1])
def test_every_count_is_positive(field: str, count: int) -> None:
    with pytest.raises(InvalidEncoderArchitectureError, match=field):
        replace(SMALL, **{field: count})


def test_the_width_is_a_multiple_of_the_heads() -> None:
    with pytest.raises(InvalidEncoderArchitectureError, match="multiple of heads"):
        replace(SMALL, width=30)


@pytest.mark.parametrize("vocabulary_size", [0, -1])
def test_a_vocabulary_has_at_least_one_channel(vocabulary_size: int) -> None:
    with pytest.raises(InvalidEncoderArchitectureError, match="vocabulary size"):
        SMALL.parameter_count(vocabulary_size)
