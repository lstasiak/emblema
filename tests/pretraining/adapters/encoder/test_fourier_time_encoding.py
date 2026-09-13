import math
from itertools import combinations

import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.fourier_time_encoding import (  # noqa: E402
    FourierTimeEncoding,
)

pytestmark = pytest.mark.ml

FREQUENCIES = 8
WIDTH = 16
# From sixteen frequencies up, the top one passes half precision's largest value, so an encoding
# this wide is where forming the angles in single precision stops being invisible.
WIDE_FREQUENCIES = 16


def test_a_timeless_token_is_encoded_as_zero() -> None:
    torch.manual_seed(1)
    encoding = FourierTimeEncoding(FREQUENCIES, WIDTH)
    timeless = torch.tensor([[True, False, True, False, False]])

    encoded = encoding(torch.rand(1, 5), timeless)

    assert torch.equal(encoded[timeless], torch.zeros(2, WIDTH))
    assert (encoded[~timeless] != 0).all(dim=-1).all()


def test_different_positions_in_the_window_get_different_encodings() -> None:
    torch.manual_seed(1)
    encoding = FourierTimeEncoding(FREQUENCIES, WIDTH)
    positions = torch.tensor([[0.0, 0.01, 0.1, 0.5, 0.9, 1.0]])

    encoded = encoding(positions, torch.zeros_like(positions, dtype=torch.bool))[0]

    for one, other in combinations(range(positions.shape[1]), 2):
        assert not torch.allclose(encoded[one], encoded[other]), (one, other)


def test_a_wide_encoding_survives_being_held_in_half_precision() -> None:
    torch.manual_seed(1)
    encoding = FourierTimeEncoding(WIDE_FREQUENCIES, WIDTH).half()
    timestamps = torch.rand(1, 5, dtype=torch.float16)

    encoded = encoding(timestamps, torch.zeros(1, 5, dtype=torch.bool))

    assert torch.isfinite(encoding.frequencies).all()
    assert torch.isfinite(encoded).all()
    assert encoded.dtype == torch.float16


def test_the_frequencies_double_from_half_a_cycle_per_window_and_are_not_learned() -> None:
    encoding = FourierTimeEncoding(FREQUENCIES, WIDTH)

    assert encoding.frequencies.tolist() == pytest.approx(
        [math.pi * 2**k for k in range(FREQUENCIES)]
    )
    assert {name for name, _ in encoding.named_parameters()} == {
        "projection.weight",
        "projection.bias",
    }
