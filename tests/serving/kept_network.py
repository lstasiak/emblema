"""A network as a campaign keeps it, for the runtime tests: its graph beside a fitted state.

Apparatus. The graph is the one the export tests stand on, over the grown vocabulary; the
fitted state is bytes nobody reads, because the runtime under test runs the graph and the
state exists only so that the manifest names a measured form.
"""

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.evaluation.adapters.features.support import timed, window
from tests.evaluation.adapters.onnx.candidates import VOCABULARY, Exported, exported
from tests.support.published import manifest_of

GRAPH_CORPUS = "graph-corpus"
# What every graph in these tests runs on: the processor, the one provider every build ships.
CPU = ("CPUExecutionProvider",)
FITTED_STATE = "torch-state"
BLOCK = ArtifactRef(key="durable/graph-block", checksum=Checksum.of_bytes(b"graph block"))


def graph_channels() -> tuple[PublishedChannel, ...]:
    """As many channels as the graph's table has rows, each fitted to leave values as they are."""
    return tuple(
        PublishedChannel(
            channel_id=channel,
            corpus=GRAPH_CORPUS,
            channel=f"c{channel}",
            statistics=PublishedChannelStatistics(1, 0.0, 1.0),
        )
        for channel in range(1, VOCABULARY + 1)
    )


def graph_corpus(store: ArtifactStore) -> ArtifactRef:
    manifest = manifest_of(BLOCK, corpus=GRAPH_CORPUS, channels=graph_channels())
    return store.put(PublishedCorpusManifestJson().encode(manifest))


def network() -> Exported:
    return exported(TransferMode.FULL_FINE_TUNING)


def keep_network(store: ArtifactStore, *, with_graph: bool = True) -> ArtifactRef:
    """The manifest of a kept network: its state, and its graph unless told otherwise."""
    derived = (
        [RepresentationBytes(InferenceGraph.FORMAT, network().graph.to_bytes(), deviation=0.0)]
        if with_graph
        else []
    )
    return KeptCandidates(store).keep(
        CandidateKind.NEURAL,
        corpus_manifest=graph_corpus(store),
        measured=RepresentationBytes(FITTED_STATE, b"the fitted state, which nothing here reads"),
        derived=derived,
    )


def network_windows(count: int) -> list[TokenWindow]:
    """``count`` windows of differing length over the first channels of the vocabulary."""
    return [
        window(
            timed(1, [(0.1 * (i + n), 0.1 * (i + 1)) for i in range(3 + n)]),
            timed(2, [(0.5, 0.25), (-0.5, 0.75)]),
            timed(3 + n, [(1.0, 0.5)]),
        )
        for n in range(count)
    ]
