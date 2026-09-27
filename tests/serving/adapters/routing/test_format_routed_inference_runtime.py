"""The routed runtime picks a form by what it can read, and never by the kind of candidate."""

import numpy as np
import pytest

pytest.importorskip("torch")
xgboost = pytest.importorskip("xgboost")

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates  # noqa: E402
from emblema.evaluation.adapters.artifacts.kept_classical_inference import (  # noqa: E402
    KeptClassicalInference,
)
from emblema.evaluation.adapters.artifacts.representation_bytes import (  # noqa: E402
    RepresentationBytes,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind  # noqa: E402
from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference  # noqa: E402
from emblema.serving.adapters.routing.format_routed_inference_runtime import (  # noqa: E402
    FormatRoutedInferenceRuntime,
)
from emblema.serving.domain.exceptions import (  # noqa: E402
    ArtifactUnavailableError,
    EmbeddingUnavailableError,
    UnreadableServedArtifactError,
    UnservableArtifactError,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from emblema.shared.kernel.retention import Retention  # noqa: E402
from tests.serving.kept_network import (  # noqa: E402
    BUDGET,
    CPU,
    gate,
    graph_channels,
    keep_network,
    network,
    network_windows,
)
from tests.serving.kept_trees import (  # noqa: E402
    fitted_trees,
    keep_trees,
    trees_answers,
    trees_corpus,
    trees_windows,
)
from tests.support.openmp import skip_if_torch_shares_the_process  # noqa: E402

pytestmark = pytest.mark.ml


class CountingStore(InMemoryArtifactStore):
    """The in-memory store, counting how often bytes are asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def get(self, ref: ArtifactRef) -> bytes:
        self.reads += 1
        return super().get(ref)


def runtime(
    store: InMemoryArtifactStore, *, graphs: bool = True, classical: bool = True
) -> FormatRoutedInferenceRuntime:
    return FormatRoutedInferenceRuntime(
        store,
        graphs=OnnxGraphInference(
            batch_size=4, providers=CPU, threads=1, budget=BUDGET, gate=gate()
        )
        if graphs
        else None,
        classical=KeptClassicalInference() if classical else None,
    )


def test_a_network_is_run_through_its_graph_and_answers_as_the_exporter_does() -> None:
    store = InMemoryArtifactStore()
    kept = keep_network(store)
    windows = network_windows(2)

    predicted = runtime(store).predict(kept, windows)
    embedded = runtime(store).embed(kept, windows)

    batch = TokenTensors.from_windows(windows)
    np.testing.assert_allclose(predicted, network().graph.predict(batch), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(embedded, network().graph.embed(batch), rtol=1e-5, atol=1e-5)
    assert runtime(store).describe(kept).channels == graph_channels()


def test_a_classical_candidate_is_run_through_the_published_service() -> None:
    skip_if_torch_shares_the_process()
    store = InMemoryArtifactStore()
    kept = keep_trees(store)
    windows = trees_windows(2)

    predicted = runtime(store).predict(kept, windows)

    assert list(predicted) == pytest.approx(trees_answers(fitted_trees(), windows))


def test_a_classical_candidate_has_no_representation_to_hand_out() -> None:
    store = InMemoryArtifactStore()
    kept = keep_trees(store)

    with pytest.raises(EmbeddingUnavailableError, match="no representation"):
        runtime(store).embed(kept, trees_windows(1))


def test_a_network_kept_without_its_graph_is_unservable_by_name() -> None:
    store = InMemoryArtifactStore()
    kept = keep_network(store, with_graph=False)

    with pytest.raises(UnservableArtifactError, match="torch-state"):
        runtime(store).predict(kept, network_windows(1))
    with pytest.raises(UnservableArtifactError):
        runtime(store).embed(kept, network_windows(1))


def test_a_process_without_the_reader_of_a_form_refuses_it_by_name() -> None:
    store = InMemoryArtifactStore()
    kept = keep_network(store)

    with pytest.raises(UnservableArtifactError, match="nothing composed in"):
        runtime(store, graphs=False, classical=False).predict(kept, network_windows(1))
    with pytest.raises(UnservableArtifactError, match="the classical forms"):
        runtime(store, graphs=False).embed(kept, network_windows(1))


def test_a_classical_form_the_service_cannot_read_is_refused_as_unreadable() -> None:
    skip_if_torch_shares_the_process()
    store = InMemoryArtifactStore()
    kept = KeptCandidates(store).keep(
        CandidateKind.CLASSICAL,
        corpus_manifest=trees_corpus(store),
        measured=RepresentationBytes("xgboost-joblib", b"not a document"),
    )

    with pytest.raises(UnreadableServedArtifactError, match="not one this reads"):
        runtime(store).predict(kept, trees_windows(1))


def test_the_manifest_and_the_form_are_read_from_the_store_once_per_process() -> None:
    store = CountingStore()
    kept = keep_network(store)
    served = runtime(store)
    served.predict(kept, network_windows(1))
    after_first = store.reads

    served.predict(kept, network_windows(1))
    served.embed(kept, network_windows(1))
    served.describe(kept)

    assert after_first == 3  # the manifest, the corpus manifest and the graph
    assert store.reads == after_first


def test_an_artifact_that_is_not_a_kept_candidate_is_refused_as_unreadable() -> None:
    store = InMemoryArtifactStore()
    kept = store.put(b"not a manifest")

    with pytest.raises(UnreadableServedArtifactError, match="not a kept candidate"):
        runtime(store).describe(kept)


def test_a_kept_candidate_whose_corpus_manifest_is_gone_is_unavailable() -> None:
    store = InMemoryArtifactStore()
    gone = ArtifactRef(key="durable/gone", checksum=Checksum.of_bytes(b"a manifest nobody stored"))
    kept = KeptCandidates(store).keep(
        CandidateKind.CLASSICAL,
        corpus_manifest=gone,
        measured=RepresentationBytes("xgboost-joblib", fitted_trees().to_bytes()),
    )

    with pytest.raises(ArtifactUnavailableError, match="gone"):
        runtime(store).describe(kept)


def test_a_transient_form_reads_like_a_durable_one() -> None:
    # Retention is the store's business; the runtime reads whatever the manifest names.
    store = InMemoryArtifactStore()
    store.put(b"anything", Retention.TRANSIENT)
    kept = keep_trees(store)

    assert runtime(store).describe(kept).corpus == "test-corpus"
