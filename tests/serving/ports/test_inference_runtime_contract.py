"""Contract of the InferenceRuntime port, run against every adapter.

The in-memory adapter runs everywhere; the routed adapter runs a real graph and real trees and
needs both stacks, so it is marked ``ml`` and skips where one is missing. Both describe what a
kept candidate takes, answer one figure per window in order, represent a network and refuse to
represent a classical candidate, and refuse an artifact nobody kept.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from emblema.serving.adapters.in_memory.inference_runtime import (
    InMemoryInferenceRuntime,
    StatedCandidate,
)
from emblema.serving.domain.exceptions import ArtifactUnavailableError, EmbeddingUnavailableError
from emblema.serving.domain.model_input import ModelInput
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from tests.evaluation.adapters.features.support import timed, window
from tests.support.openmp import skip_if_torch_shares_the_process
from tests.support.published import CHANNELS, CORPUS

NOBODY_KEPT = ArtifactRef(key="durable/nobody", checksum=Checksum.of_bytes(b"nobody"))


@dataclass(frozen=True)
class Case:
    """A runtime with a network and a classical candidate kept for it, and windows they take."""

    runtime: InferenceRuntime
    network: ArtifactRef
    classical: ArtifactRef
    network_windows: list[TokenWindow]
    classical_windows: list[TokenWindow]
    network_corpus: str
    classical_corpus: str


def in_memory() -> Case:
    network = ArtifactRef(key="durable/network", checksum=Checksum.of_bytes(b"network"))
    classical = ArtifactRef(key="durable/trees", checksum=Checksum.of_bytes(b"trees"))
    windows = [window(timed(1, [(0.5, 0.5)]))]
    runtime = InMemoryInferenceRuntime(
        {
            network.checksum: StatedCandidate(
                input=ModelInput(corpus="graphs", window_length=1.0, channels=CHANNELS),
                prediction=3.0,
                embedding=(0.5, 0.25),
            ),
            classical.checksum: StatedCandidate(
                input=ModelInput(corpus=CORPUS, window_length=10.0, channels=CHANNELS),
                prediction=4.0,
            ),
        }
    )
    return Case(runtime, network, classical, windows, windows, "graphs", CORPUS)


def format_routed() -> Case:
    pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    skip_if_torch_shares_the_process()
    from emblema.evaluation.adapters.artifacts.kept_classical_inference import (
        KeptClassicalInference,
    )
    from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference
    from emblema.serving.adapters.routing.format_routed_inference_runtime import (
        FormatRoutedInferenceRuntime,
    )
    from tests.serving.kept_network import (
        BUDGET,
        CPU,
        GRAPH_CORPUS,
        gate,
        keep_network,
        network_windows,
    )
    from tests.serving.kept_trees import keep_trees, trees_windows

    store = InMemoryArtifactStore()
    runtime = FormatRoutedInferenceRuntime(
        store,
        graphs=OnnxGraphInference(
            batch_size=4, providers=CPU, threads=1, budget=BUDGET, gate=gate()
        ),
        classical=KeptClassicalInference(),
    )
    return Case(
        runtime,
        keep_network(store),
        keep_trees(store),
        network_windows(3),
        trees_windows(3),
        GRAPH_CORPUS,
        CORPUS,
    )


ADAPTERS: dict[str, Callable[[], Case]] = {"in_memory": in_memory, "format_routed": format_routed}


@pytest.fixture(
    params=[
        pytest.param("in_memory", id="in_memory"),
        pytest.param("format_routed", id="format_routed", marks=pytest.mark.ml),
    ]
)
def case(request: pytest.FixtureRequest) -> Case:
    return ADAPTERS[request.param]()


def test_a_kept_candidate_is_described_by_the_corpus_it_was_fitted_to(case: Case) -> None:
    assert case.runtime.describe(case.network).corpus == case.network_corpus
    assert case.runtime.describe(case.classical).corpus == case.classical_corpus
    assert case.runtime.describe(case.classical).channels == CHANNELS


def test_a_network_answers_one_figure_per_window_in_order(case: Case) -> None:
    answered = case.runtime.predict(case.network, case.network_windows)

    assert len(answered) == len(case.network_windows)
    assert all(isinstance(answer, float) for answer in answered)


def test_a_classical_candidate_answers_one_figure_per_window_in_order(case: Case) -> None:
    answered = case.runtime.predict(case.classical, case.classical_windows)

    assert len(answered) == len(case.classical_windows)
    assert all(isinstance(answer, float) for answer in answered)


def test_a_network_represents_every_window_with_the_same_width(case: Case) -> None:
    embedded = case.runtime.embed(case.network, case.network_windows)

    assert len(embedded) == len(case.network_windows)
    assert len({len(row) for row in embedded}) == 1


def test_a_classical_candidate_has_no_representation_to_hand_out(case: Case) -> None:
    with pytest.raises(EmbeddingUnavailableError):
        case.runtime.embed(case.classical, case.classical_windows)


def test_an_artifact_nobody_kept_is_refused(case: Case) -> None:
    with pytest.raises(ArtifactUnavailableError):
        case.runtime.describe(NOBODY_KEPT)
    with pytest.raises(ArtifactUnavailableError):
        case.runtime.predict(NOBODY_KEPT, case.network_windows)
