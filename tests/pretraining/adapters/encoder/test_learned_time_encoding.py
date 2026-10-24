import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.fourier_time_encoding import (  # noqa: E402
    FourierTimeEncoding,
)
from emblema.pretraining.adapters.encoder.learned_time_encoding import (  # noqa: E402
    LearnedTimeEncoding,
)

pytestmark = pytest.mark.ml

FREQUENCIES = 8
WIDTH = 16
WIDE_FREQUENCIES = 16


def encodings() -> tuple[FourierTimeEncoding, LearnedTimeEncoding]:
    torch.manual_seed(1)
    fixed = FourierTimeEncoding(FREQUENCIES, WIDTH)
    learnt = LearnedTimeEncoding(FREQUENCIES, WIDTH)
    learnt.projection.load_state_dict(fixed.projection.state_dict())
    return fixed, learnt


def test_at_the_start_the_learnt_encoding_is_the_fixed_one_bit_for_bit() -> None:
    fixed, learnt = encodings()
    timestamps = torch.rand(3, 7)
    timeless = torch.rand(3, 7) < 0.3

    assert torch.equal(learnt(timestamps, timeless), fixed(timestamps, timeless))
    assert torch.equal(learnt.frequencies, fixed.frequencies)


def test_the_frequencies_and_the_phases_learn_and_the_fixed_ones_do_not() -> None:
    fixed, learnt = encodings()
    timestamps = torch.rand(2, 5)
    timeless = torch.zeros(2, 5, dtype=torch.bool)

    learnt(timestamps, timeless).sum().backward()

    for parameter in (learnt.octave_shift, learnt.phases):
        assert parameter.grad is not None
        assert (parameter.grad != 0).any()
    assert "octaves" not in dict(fixed.named_parameters())
    assert {name for name, _ in learnt.named_parameters()} == {
        "octave_shift",
        "phases",
        "projection.weight",
        "projection.bias",
    }


def test_the_learnt_parts_are_zero_so_a_decay_pulls_towards_the_fixed_encoding() -> None:
    _, learnt = encodings()

    assert torch.equal(learnt.octave_shift, torch.zeros(FREQUENCIES))
    assert torch.equal(learnt.phases, torch.zeros(FREQUENCIES))
    assert torch.equal(learnt.octaves, torch.arange(FREQUENCIES, dtype=torch.float32))


def test_a_timeless_token_is_encoded_as_zero_whatever_the_phases() -> None:
    _, learnt = encodings()
    with torch.no_grad():
        learnt.phases.add_(0.7)
    timeless = torch.tensor([[True, False, True]])

    encoded = learnt(torch.rand(1, 3), timeless)

    assert torch.equal(encoded[timeless], torch.zeros(2, WIDTH))
    assert (encoded[~timeless] != 0).all()


def test_a_wide_encoding_survives_being_held_in_half_precision() -> None:
    torch.manual_seed(2)
    learnt = LearnedTimeEncoding(WIDE_FREQUENCIES, WIDTH).half()

    encoded = learnt(torch.rand(2, 4).half(), torch.zeros(2, 4, dtype=torch.bool))

    assert encoded.dtype == torch.float16
    assert torch.isfinite(encoded).all()
