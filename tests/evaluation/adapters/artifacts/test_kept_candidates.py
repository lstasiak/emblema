"""A candidate kept in several forms: every form in the store, one manifest naming them."""

import pytest

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.exceptions import (
    InvalidKeptCandidateManifestError,
    MalformedKeptCandidateManifestError,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

CORPUS = ArtifactRef(key="durable/corpus", checksum=Checksum.of_bytes(b"corpus"))
STATE = RepresentationBytes("torch-state", b"the weights")
GRAPH = RepresentationBytes("onnx", b"the graph", deviation=2e-5)


def test_every_form_lands_in_the_store_and_the_manifest_names_them() -> None:
    store = InMemoryArtifactStore()
    kept = KeptCandidates(store)

    manifest = kept.keep(
        CandidateKind.NEURAL, corpus_manifest=CORPUS, measured=STATE, derived=(GRAPH,)
    )

    read = kept.read(manifest)
    assert read.kind is CandidateKind.NEURAL
    assert read.corpus_manifest == CORPUS
    assert read.measured_as == "torch-state"
    assert store.get(read.measured.artifact) == b"the weights"
    graph = read.find("onnx")
    assert graph is not None
    assert store.get(graph.artifact) == b"the graph"
    assert graph.deviation == pytest.approx(2e-5)


def test_the_same_forms_kept_twice_are_one_manifest() -> None:
    kept = KeptCandidates(InMemoryArtifactStore())

    first = kept.keep(CandidateKind.CLASSICAL, corpus_manifest=CORPUS, measured=STATE)
    second = kept.keep(CandidateKind.CLASSICAL, corpus_manifest=CORPUS, measured=STATE)

    assert first == second


def test_a_derived_form_without_a_deviation_is_refused() -> None:
    kept = KeptCandidates(InMemoryArtifactStore())

    with pytest.raises(InvalidKeptCandidateManifestError):
        kept.keep(
            CandidateKind.NEURAL,
            corpus_manifest=CORPUS,
            measured=STATE,
            derived=(RepresentationBytes("onnx", b"the graph"),),
        )


def test_bytes_that_are_not_a_manifest_are_refused_when_read() -> None:
    store = InMemoryArtifactStore()

    with pytest.raises(MalformedKeptCandidateManifestError):
        KeptCandidates(store).read(store.put(b"the weights"))
