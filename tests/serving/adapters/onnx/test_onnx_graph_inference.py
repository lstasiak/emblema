"""The graph reader Serving runs agrees with the exporter that wrote the graph."""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("torch")

from emblema.evaluation.contracts.kept_representation import KeptRepresentation  # noqa: E402
from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference  # noqa: E402
from emblema.serving.adapters.onnx.weighted_semaphore import WeightedSemaphore  # noqa: E402
from emblema.serving.domain.exceptions import (  # noqa: E402
    InferenceBusyError,
    UnreadableServedArtifactError,
    WindowBeyondBudgetError,
)
from emblema.serving.domain.inference_budget import InferenceBudget  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from tests.serving.kept_network import (  # noqa: E402
    BUDGET,
    CPU,
    gate,
    graph_reader,
    network,
    network_windows,
)

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
    reader = graph_reader(8)

    predicted = reader.predict(graph_form(content), content, windows)
    embedded = reader.embed(graph_form(content), content, windows)

    batch = TokenTensors.from_windows(windows)
    np.testing.assert_allclose(predicted, exported.graph.predict(batch), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(embedded, exported.graph.embed(batch), rtol=1e-5, atol=1e-5)


def test_windows_answered_in_batches_are_answered_as_one() -> None:
    content = network().graph.to_bytes()
    windows = network_windows(5)

    one = OnnxGraphInference(
        batch_size=1, providers=CPU, threads=1, budget=BUDGET, gate=gate()
    ).predict(graph_form(content), content, windows)
    whole = OnnxGraphInference(
        batch_size=16, providers=CPU, threads=1, budget=BUDGET, gate=gate()
    ).predict(graph_form(content), content, windows)

    np.testing.assert_allclose(one, whole, rtol=1e-5, atol=1e-4)


def test_bytes_that_are_not_a_graph_are_refused() -> None:
    with pytest.raises(UnreadableServedArtifactError, match="not a graph"):
        OnnxGraphInference(
            batch_size=4, providers=CPU, threads=1, budget=BUDGET, gate=gate()
        ).predict(graph_form(b"nonsense"), b"nonsense", network_windows(1))


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
        OnnxGraphInference(
            batch_size=4, providers=CPU, threads=1, budget=BUDGET, gate=gate()
        ).predict(graph_form(content), content, network_windows(1))


def test_a_batch_holds_at_least_one_window() -> None:
    with pytest.raises(ValueError, match="at least one"):
        graph_reader(0)


def test_a_graph_runs_on_a_provider_and_a_thread_it_was_given() -> None:
    with pytest.raises(ValueError, match="execution provider"):
        OnnxGraphInference(batch_size=1, providers=(), threads=1, budget=BUDGET, gate=gate())
    with pytest.raises(ValueError, match="thread"):
        OnnxGraphInference(batch_size=1, providers=CPU, threads=0, budget=BUDGET, gate=gate())


class CountingSemaphore(WeightedSemaphore):
    """A semaphore that remembers what was reserved, batch by batch, and until when."""

    def __init__(self, capacity: int, *, wait_seconds: float) -> None:
        super().__init__(capacity, wait_seconds=wait_seconds)
        self.reserved: list[int] = []
        self.deadlines: list[float | None] = []

    @contextmanager
    def reserve(self, weight: int, *, until: float | None = None) -> Iterator[None]:
        self.reserved.append(weight)
        self.deadlines.append(until)
        with super().reserve(weight, until=until):
            yield


def reader_under(budget: InferenceBudget, gate: WeightedSemaphore) -> OnnxGraphInference:
    return OnnxGraphInference(batch_size=16, providers=CPU, threads=1, budget=budget, gate=gate)


def test_a_request_is_cut_by_what_a_batch_costs_and_answered_as_if_it_were_not() -> None:
    content = network().graph.to_bytes()
    windows = network_windows(5)  # 6 to 10 tokens each: 10, 9, 8 alone, 7 and 6 together
    budget = InferenceBudget(windows=1, longest=10)
    counting = CountingSemaphore(budget.capacity, wait_seconds=5)

    cut = reader_under(budget, counting).predict(graph_form(content), content, windows)
    whole = graph_reader(16).predict(graph_form(content), content, windows)

    assert counting.reserved == [100, 81, 64, 98]
    np.testing.assert_allclose(cut, whole, rtol=1e-5, atol=1e-4)


def test_every_batch_of_a_request_waits_under_the_one_deadline_the_request_was_given() -> None:
    content = network().graph.to_bytes()
    budget = InferenceBudget(windows=1, longest=10)
    counting = CountingSemaphore(budget.capacity, wait_seconds=5)

    reader_under(budget, counting).predict(graph_form(content), content, network_windows(5))

    assert len(counting.deadlines) == 4
    assert len(set(counting.deadlines)) == 1
    assert counting.deadlines[0] is not None


def test_a_reader_answers_in_the_order_it_was_asked_whatever_order_it_ran_in() -> None:
    content = network().graph.to_bytes()
    windows = network_windows(5)
    reader = graph_reader(16)

    forward = reader.predict(graph_form(content), content, windows)
    backward = reader.predict(graph_form(content), content, windows[::-1])

    np.testing.assert_allclose(forward, backward[::-1], rtol=1e-5, atol=1e-4)


def test_a_request_that_cannot_be_admitted_in_time_is_refused_and_frees_what_it_held() -> None:
    content = network().graph.to_bytes()
    windows = network_windows(2)
    budget = InferenceBudget(windows=1, longest=10)
    busy = WeightedSemaphore(budget.capacity, wait_seconds=0.05)

    with busy.reserve(budget.capacity), pytest.raises(InferenceBusyError):
        reader_under(budget, busy).predict(graph_form(content), content, windows)

    assert busy.in_use == 0
    reader_under(budget, busy).predict(graph_form(content), content, windows)


def test_a_window_longer_than_the_budget_admits_is_refused_before_it_is_run() -> None:
    content = network().graph.to_bytes()
    budget = InferenceBudget(windows=1, longest=8)
    counting = CountingSemaphore(budget.capacity, wait_seconds=5)

    with pytest.raises(WindowBeyondBudgetError):
        reader_under(budget, counting).predict(graph_form(content), content, network_windows(5))

    assert counting.reserved == []


def test_a_semaphore_that_holds_more_or_less_than_the_budget_is_refused() -> None:
    budget = InferenceBudget(windows=2, longest=10)

    with pytest.raises(ValueError, match="holds 50"):
        reader_under(budget, WeightedSemaphore(50, wait_seconds=1))


def test_what_a_run_allocated_is_given_back_after_it() -> None:
    options = graph_reader().run_options

    assert options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "cpu:0"
