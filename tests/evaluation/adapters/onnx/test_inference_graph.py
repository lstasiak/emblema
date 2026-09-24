"""Does a fitted candidate, with one dynamic axis, survive the round trip to ONNX Runtime?

The question was first answered on a stand-in, then on the bare encoder; from here the graph is
the candidate a campaign keeps — encoder, pooling and head, under every transfer mode, over a
vocabulary the task grew — so a construct the exporter cannot trace fails on the commit that
introduces it, not when a campaign first keeps a candidate.
"""

import numpy as np
import onnx
import pytest
from torch import Tensor

from emblema.evaluation.adapters.onnx.inference_graph import (
    INPUT_NAMES,
    MAX_TOKENS,
    OUTPUT_NAMES,
    InferenceGraph,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.domain.exceptions import (
    InferenceGraphDivergedError,
    UnexportableCandidateError,
    UnreadableInferenceGraphError,
)
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import N_FEATURES
from tests.evaluation.adapters.onnx.candidates import (
    TARGET_SCALE,
    Exported,
    adapted,
    exported,
    over_grown_channels,
)
from tests.support.experiments import windows
from tests.support.token_tensors import (
    all_timeless,
    fully_padded,
    padded_by,
    permuted,
    random_batch,
    with_timestamps,
)

pytestmark = [
    pytest.mark.ml,
    # The exporter reports that it dropped one of five identical axis names. It kept the name,
    # and a test below asserts that; the notice is about its own renaming pass.
    pytest.mark.filterwarnings("ignore:# The axis name:UserWarning"),
]

# float32 through a different set of kernels. The largest deviation observed on the pooled state
# across every case below and both development machines is 3.6e-06, at the longest window, where
# the reduction accumulates the most. The tolerance keeps a factor of three over that; the answer
# is that state through a linear head and multiplied by the ceiling, so its tolerance scales.
ATOL = 1e-5
RTOL = 1e-4


def assert_matches_eager(pair: Exported, batch: TokenTensors) -> None:
    np.testing.assert_allclose(
        pair.graph.embed(batch), pair.embed_eager(batch), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        pair.graph.predict(batch), pair.predict_eager(batch), rtol=RTOL, atol=ATOL * TARGET_SCALE
    )


def assert_same_answers(graph: InferenceGraph, one: TokenTensors, other: TokenTensors) -> None:
    np.testing.assert_allclose(graph.embed(one), graph.embed(other), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        graph.predict(one), graph.predict(other), rtol=RTOL, atol=ATOL * TARGET_SCALE
    )


def test_each_input_name_belongs_to_the_argument_it_is_exported_beside() -> None:
    # The export pairs names with arguments by position, so a name that slid one place would
    # label every tensor in the graph wrongly while the graph still ran.
    batch = random_batch(1, 4, seed=1)

    named = tuple(getattr(batch, name) for name in INPUT_NAMES)

    assert all(one is other for one, other in zip(named, batch.args, strict=True))


def test_the_graph_takes_the_five_named_tensors_and_returns_the_state_and_the_answer(
    lora: Exported,
) -> None:
    graph = lora.graph

    assert graph.input_names == INPUT_NAMES
    assert graph.output_names == OUTPUT_NAMES
    assert graph.get_input_shape("features") == ("batch", "n_tokens", N_FEATURES)
    for name in ("channel_ids", "timestamps", "timeless", "padding_mask"):
        assert graph.get_input_shape(name) == ("batch", "n_tokens")
    width = lora.candidate.head.linear.in_features
    assert graph.get_output_shape("pooled_embedding") == ("batch", width)
    assert graph.get_output_shape("prediction") == ("batch",)


def test_the_token_count_is_the_only_dynamic_axis_beyond_the_batch(lora: Exported) -> None:
    # A third symbolic axis would mean a channel-by-time grid leaked into the representation,
    # which is exactly the assumption the encoder must not make about sampling.
    assert lora.graph.symbolic_axes == {"batch", "n_tokens"}


@pytest.mark.parametrize("mode", list(TransferMode))
@pytest.mark.parametrize("tokens", [137, 512])
def test_under_every_mode_both_outputs_match_pytorch_at_different_token_counts(
    mode: TransferMode, tokens: int
) -> None:
    assert_matches_eager(exported(mode), random_batch(1, tokens, seed=tokens))


@pytest.mark.parametrize("mode", list(TransferMode))
def test_under_every_mode_a_window_over_the_tasks_grown_channels_matches_pytorch(
    mode: TransferMode,
) -> None:
    # A pretrained candidate reads the grown channels from a second table through a select; a
    # candidate from scratch has them in its one table. Either way the graph has to agree.
    assert_matches_eager(exported(mode), over_grown_channels(random_batch(2, 64, seed=9)))


def test_outputs_match_pytorch_for_a_batch_of_partly_padded_windows(lora: Exported) -> None:
    assert_matches_eager(lora, random_batch(3, 41, seed=41, padding=17))


def test_padding_does_not_change_the_outputs(lora: Exported) -> None:
    batch = random_batch(1, 64, seed=13)

    assert_same_answers(lora.graph, padded_by(batch, 32), batch)


def test_reordering_the_tokens_does_not_change_the_outputs(lora: Exported) -> None:
    batch = random_batch(1, 64, seed=11)

    assert_same_answers(lora.graph, permuted(batch, seed=3), batch)


def test_a_timeless_token_ignores_its_timestamp(lora: Exported) -> None:
    batch = all_timeless(random_batch(1, 32, seed=5))

    assert_same_answers(lora.graph, with_timestamps(batch, seed=6), batch)


def test_a_window_of_nothing_but_padding_yields_finite_values(lora: Exported) -> None:
    # Attention over an entirely masked window is a softmax over nothing. The attention was
    # chosen for turning it into zeros rather than NaN; the exported graph must keep that guard.
    batch = fully_padded(random_batch(1, 16, seed=7))

    assert np.isfinite(lora.graph.embed(batch)).all()
    assert np.isfinite(lora.graph.predict(batch)).all()


def test_more_tokens_than_the_export_declared_still_run(lora: Exported) -> None:
    # The declared upper bound guides the exporter; the runtime does not enforce it. Rejecting
    # an oversized window is therefore the caller's job, not the graph's.
    assert_matches_eager(lora, random_batch(1, MAX_TOKENS + 8, seed=1))


class Untraceable(AdaptedBackbone):
    """A candidate whose answer branches on a value the exporter cannot know while tracing."""

    def embed(self, batch: TokenTensors) -> Tensor:
        if float(batch.features.sum()) > 0.0:
            return super().embed(batch)
        return super().embed(batch) * 2


def test_a_candidate_the_exporter_cannot_trace_is_refused_with_a_domain_error() -> None:
    source = adapted(TransferMode.FULL_FINE_TUNING)
    candidate = Untraceable(source.encoder, source.head)

    with pytest.raises(UnexportableCandidateError, match="does not export"):
        InferenceGraph.exported(candidate, target_scale=TARGET_SCALE)


def test_a_graph_comes_back_from_its_bytes_as_it_went_in(lora: Exported) -> None:
    batch = random_batch(2, 50, seed=50)

    read = InferenceGraph.read(lora.graph.to_bytes())

    assert read.symbolic_axes == lora.graph.symbolic_axes
    np.testing.assert_array_equal(read.predict(batch), lora.graph.predict(batch))


@pytest.mark.parametrize(
    "raw",
    [
        b"not onnx",
        b"\xff\xfe",
        b"",
        onnx.helper.make_model(onnx.helper.make_graph([], "other", [], [])).SerializeToString(),
    ],
    ids=["text", "binary noise", "nothing", "another graph"],
)
def test_bytes_that_are_not_an_inference_graph_are_refused(raw: bytes) -> None:
    with pytest.raises(UnreadableInferenceGraphError):
        InferenceGraph.read(raw)


def test_the_deviation_from_the_answers_the_graph_was_derived_from_is_reported(
    lora: Exported,
) -> None:
    # The check a runtime makes before it keeps a graph: the measured answers on the windows the
    # campaign scored, against the graph's own, in the task's unit.
    scored = windows(5, seed=3)
    measured = lora.predict_eager(TokenTensors.from_windows(scored)).tolist()

    deviation = lora.graph.deviation_from(measured, scored, target_scale=TARGET_SCALE, batch_size=2)

    assert 0.0 <= deviation <= ATOL * TARGET_SCALE


def test_a_graph_that_strays_from_the_measured_answers_is_refused(lora: Exported) -> None:
    scored = windows(3, seed=4)
    other = (lora.predict_eager(TokenTensors.from_windows(scored)) + TARGET_SCALE).tolist()

    with pytest.raises(InferenceGraphDivergedError, match="strays"):
        lora.graph.deviation_from(other, scored, target_scale=TARGET_SCALE, batch_size=8)
