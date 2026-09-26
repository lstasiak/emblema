"""The graph reader Serving runs agrees with the exporter that wrote the graph."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("torch")

from emblema.evaluation.contracts.kept_representation import KeptRepresentation  # noqa: E402
from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference  # noqa: E402
from emblema.serving.domain.exceptions import UnreadableServedArtifactError  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from tests.serving.kept_network import CPU, network, network_windows  # noqa: E402

pytestmark = pytest.mark.ml


def graph_form(content: bytes) -> KeptRepresentation:
    return KeptRepresentation(
        format=OnnxGraphInference.FORMAT,
        artifact=ArtifactRef(key="durable/graph", checksum=Checksum.of_bytes(content)),
        deviation=0.0,
    )


def test_the_reader_answers_what_the_exporter_answers_for_the_same_windows() -> None:
    exported = network()
    windows = network_windows(3)
    content = exported.graph.to_bytes()
    reader = OnnxGraphInference(batch_size=8, providers=CPU, threads=1)

    predicted = reader.predict(graph_form(content), content, windows)
    embedded = reader.embed(graph_form(content), content, windows)

    batch = TokenTensors.from_windows(windows)
    np.testing.assert_allclose(predicted, exported.graph.predict(batch), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(embedded, exported.graph.embed(batch), rtol=1e-5, atol=1e-5)


def test_windows_answered_in_batches_are_answered_as_one() -> None:
    content = network().graph.to_bytes()
    windows = network_windows(5)

    one = OnnxGraphInference(batch_size=1, providers=CPU, threads=1).predict(
        graph_form(content), content, windows
    )
    whole = OnnxGraphInference(batch_size=16, providers=CPU, threads=1).predict(
        graph_form(content), content, windows
    )

    np.testing.assert_allclose(one, whole, rtol=1e-5, atol=1e-4)


def test_bytes_that_are_not_a_graph_are_refused() -> None:
    with pytest.raises(UnreadableServedArtifactError, match="not a graph"):
        OnnxGraphInference(batch_size=4, providers=CPU, threads=1).predict(
            graph_form(b"nonsense"), b"nonsense", network_windows(1)
        )


def test_a_graph_of_another_signature_is_refused() -> None:
    proto = onnx.ModelProto()
    proto.CopyFrom(network().graph.proto)
    proto.graph.output[1].name = "answer"
    for node in proto.graph.node:
        for index, name in enumerate(node.output):
            if name == "prediction":
                node.output[index] = "answer"
    content = proto.SerializeToString()

    with pytest.raises(UnreadableServedArtifactError, match="not an inference graph of ours"):
        OnnxGraphInference(batch_size=4, providers=CPU, threads=1).predict(
            graph_form(content), content, network_windows(1)
        )


def test_a_batch_holds_at_least_one_window() -> None:
    with pytest.raises(ValueError, match="at least one"):
        OnnxGraphInference(batch_size=0, providers=CPU, threads=1)


def test_a_graph_runs_on_a_provider_and_a_thread_it_was_given() -> None:
    with pytest.raises(ValueError, match="execution provider"):
        OnnxGraphInference(batch_size=1, providers=(), threads=1)
    with pytest.raises(ValueError, match="thread"):
        OnnxGraphInference(batch_size=1, providers=CPU, threads=0)
