"""What the report times is machine-dependent; what it says about a window length is not.

The verdict — whether full attention over a corpus's default window fits the published tier's
device — is the line the report exists to print, so it is held here to the arithmetic it rests on,
on any machine. The measurement itself is only checked to run and to report what a CPU can report.
"""

import pytest

pytest.importorskip("torch")

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.shared.kernel.compute import ComputeTier
from scripts.budget_file import vocabulary_size
from scripts.encoder_budget_report import (
    DEVICE_GIB,
    GIB,
    HEADROOM_GIB,
    Measurements,
    Step,
    attention_bytes,
    batch_for,
    fits_published_device,
    measure_step,
    render,
    window_lengths,
)

pytestmark = pytest.mark.ml

SMALL = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=4)


def test_the_default_window_of_cmapss_holds_the_tokens_the_data_spike_counted() -> None:
    cmapss = next(length for length in window_lengths() if length.corpus == "cmapss")

    assert (cmapss.window, cmapss.tokens) == ("w50s5", 1050)


def test_window_lengths_come_shortest_first() -> None:
    tokens = [length.tokens for length in window_lengths()]

    assert tokens == sorted(tokens)
    assert len(tokens) >= 2


def test_attention_holds_two_square_matrices_per_head_and_layer() -> None:
    # 2 matrices × 2 heads × 1 layer × 10² tokens × 4 bytes
    assert attention_bytes(SMALL, 10, bytes_per_value=4) == 1_600


def test_the_verdict_flips_where_the_buffers_exceed_the_headroom() -> None:
    published = architecture_of(ComputeTiers.load().profile(ComputeTier.M))
    budget = (DEVICE_GIB - HEADROOM_GIB) * GIB

    fitting = 1050
    too_long = 16_000

    assert fits_published_device(published, fitting)
    assert not fits_published_device(published, too_long)
    assert 8 * attention_bytes(published, fitting, bytes_per_value=2) <= budget
    assert 8 * attention_bytes(published, too_long, bytes_per_value=2) > budget


def test_a_step_always_carries_at_least_one_window() -> None:
    assert batch_for(1_000_000) == 1
    assert batch_for(1024) == 8


def test_a_step_on_the_cpu_is_timed_and_reports_no_memory() -> None:
    step = measure_step(SMALL, tokens=16, batch=2, device="cpu")

    assert step.seconds > 0.0
    assert step.bytes_held is None


def test_the_vocabulary_spans_every_measured_corpus() -> None:
    assert vocabulary_size() > 21  # more than C-MAPSS alone


def test_the_report_states_a_verdict_per_window_length() -> None:
    lengths = window_lengths()
    architectures = {"S": SMALL, "M": architecture_of(ComputeTiers.load().profile(ComputeTier.M))}
    steps = {
        (name, length.tokens): Step(0.25, None) for name in architectures for length in lengths
    }

    report = render(Measurements("cpu", lengths, architectures, steps))

    assert report.count("— fits") + report.count("— DOES NOT FIT") == len(lengths)
    assert "### Verdict" in report
