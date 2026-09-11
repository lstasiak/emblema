"""Can a set-shaped encoder with one dynamic axis survive the round trip to ONNX Runtime?

Answering this while the architecture is still open is the point: finding it out once the encoder is
frozen would leave only two options, rewriting the model or abandoning ONNX. From here on the suite
also guards the answer, until the real export replaces it.
"""

import numpy as np
import pytest

from emblema.shared.adapters.arrays.token_batch import N_FEATURES
from tests.ml.onnx_export.batches import (
    INPUT_NAMES,
    all_timeless,
    fully_padded,
    padded_by,
    permuted,
    random_batch,
    with_timestamps,
)
from tests.ml.onnx_export.exported_encoder import (
    MAX_TOKENS,
    OUTPUT_NAME,
    ExportedEncoder,
)

pytestmark = [
    pytest.mark.ml,
    # The exporter reports that it dropped one of five identical axis names. It kept the name, and
    # the first test below asserts that; the notice is about its own renaming pass.
    pytest.mark.filterwarnings("ignore:# The axis name:UserWarning"),
]

# float32 through a different set of kernels. The largest deviation observed across the three
# attention implementations, every case below and both development machines is 3.6e-06, at the
# longest window, where the reduction accumulates the most. The tolerance keeps a factor of three
# over that.
ATOL = 1e-5
RTOL = 1e-4


def test_each_input_name_belongs_to_the_argument_it_is_exported_beside() -> None:
    # The export pairs names with arguments by position, so a name that slid one place would label
    # every tensor in the graph wrongly while the graph still ran.
    batch = random_batch(1, 4, seed=1)

    named = tuple(getattr(batch, name) for name in INPUT_NAMES)

    assert all(one is other for one, other in zip(named, batch.args, strict=True))


def test_the_graph_takes_the_five_named_tensors(exported: ExportedEncoder) -> None:
    assert exported.get_input_shape("features") == ("batch", "n_tokens", N_FEATURES)
    for name in ("channel_ids", "timestamps", "timeless", "padding_mask"):
        assert exported.get_input_shape(name) == ("batch", "n_tokens")
    assert exported.output_shape == ("batch", exported.model.d_model)
    assert exported.graph.graph.output[0].name == OUTPUT_NAME


def test_the_token_count_is_the_only_dynamic_axis_beyond_the_batch(
    exported: ExportedEncoder,
) -> None:
    # A third symbolic axis would mean a channel-by-time grid leaked into the representation, which
    # is exactly the assumption the encoder must not make about sampling.
    assert exported.symbolic_axes == {"batch", "n_tokens"}


@pytest.mark.parametrize("tokens", [137, 512])
def test_output_matches_pytorch_at_different_token_counts(
    exported: ExportedEncoder, tokens: int
) -> None:
    batch = random_batch(1, tokens, seed=tokens)

    np.testing.assert_allclose(
        exported.run_onnx(batch), exported.run_eager(batch), rtol=RTOL, atol=ATOL
    )


def test_output_matches_pytorch_for_a_batch_of_partly_padded_windows(
    exported: ExportedEncoder,
) -> None:
    batch = random_batch(3, 41, seed=41, padding=17)

    np.testing.assert_allclose(
        exported.run_onnx(batch), exported.run_eager(batch), rtol=RTOL, atol=ATOL
    )


def test_padding_does_not_change_the_embedding(exported: ExportedEncoder) -> None:
    batch = random_batch(1, 64, seed=13)

    np.testing.assert_allclose(
        exported.run_onnx(padded_by(batch, 32)), exported.run_onnx(batch), rtol=RTOL, atol=ATOL
    )


def test_reordering_the_tokens_does_not_change_the_embedding(exported: ExportedEncoder) -> None:
    batch = random_batch(1, 64, seed=11)

    np.testing.assert_allclose(
        exported.run_onnx(permuted(batch, seed=3)), exported.run_onnx(batch), rtol=RTOL, atol=ATOL
    )


def test_a_timeless_token_ignores_its_timestamp(exported: ExportedEncoder) -> None:
    batch = all_timeless(random_batch(1, 32, seed=5))

    np.testing.assert_allclose(
        exported.run_onnx(with_timestamps(batch, seed=6)),
        exported.run_onnx(batch),
        rtol=RTOL,
        atol=ATOL,
    )


def test_a_window_of_nothing_but_padding_yields_finite_values(
    exported_with_sdpa: ExportedEncoder,
) -> None:
    # Attention over an entirely masked window is a softmax over nothing. The exported graph keeps
    # the guard that turns it into zeros; the two mask-by-negative-infinity implementations do not,
    # which is why only this one is held to it.
    embedding = exported_with_sdpa.run_onnx(fully_padded(random_batch(1, 16, seed=7)))

    assert np.isfinite(embedding).all()


def test_more_tokens_than_the_export_declared_still_run(
    exported_with_sdpa: ExportedEncoder,
) -> None:
    # The declared upper bound guides the exporter; the runtime does not enforce it. Rejecting an
    # oversized window is therefore the caller's job, not the graph's.
    batch = random_batch(1, MAX_TOKENS + 8, seed=1)

    np.testing.assert_allclose(
        exported_with_sdpa.run_onnx(batch),
        exported_with_sdpa.run_eager(batch),
        rtol=RTOL,
        atol=ATOL,
    )
