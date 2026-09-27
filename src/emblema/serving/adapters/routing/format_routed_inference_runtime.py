from collections.abc import Sequence
from dataclasses import dataclass

from emblema.catalog.contracts.exceptions import CatalogContractError
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.contracts.exceptions import EvaluationContractError
from emblema.evaluation.contracts.kept_candidate_inference import KeptCandidateInference
from emblema.evaluation.contracts.kept_candidate_manifest import KeptCandidateManifest
from emblema.evaluation.contracts.kept_candidate_manifest_json import KeptCandidateManifestJson
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.serving.adapters.onnx.onnx_graph_inference import OnnxGraphInference
from emblema.serving.domain.exceptions import (
    ArtifactUnavailableError,
    EmbeddingUnavailableError,
    InvalidModelInputError,
    UnreadableServedArtifactError,
    UnservableArtifactError,
)
from emblema.serving.domain.model_input import ModelInput
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.exceptions import ArtifactStoreError


@dataclass(frozen=True)
class _KeptCandidate:
    manifest: KeptCandidateManifest
    input: ModelInput


class FormatRoutedInferenceRuntime:
    """Runs whichever form of a kept candidate this process has a reader for.

    Routed by the format the manifest names and never by the kind of candidate: the kind says
    what a candidate is made of, the format says what its bytes are, and it is the bytes that
    are run. A network is run through its graph, where one was kept and this process reads
    graphs; a classical candidate through the service the Evaluation context publishes for its
    fitted forms, where that was composed in. A candidate kept in no form this process reads —
    a network whose graph was never derived, a form of a later date — is refused by name.

    Each may be absent, so that the process serving is composed with only the stacks it needs.
    Manifests, the inputs read off them and the bytes of the forms run are kept by checksum for
    the life of the process, since a service answers many requests with the same few candidates.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        graphs: OnnxGraphInference | None = None,
        classical: KeptCandidateInference | None = None,
    ) -> None:
        self._store = store
        self._graphs = graphs
        self._classical = classical
        self._manifests = KeptCandidateManifestJson()
        self._corpora = PublishedCorpusManifestJson()
        self._candidates: dict[Checksum, _KeptCandidate] = {}
        self._contents: dict[Checksum, bytes] = {}

    def describe(self, artifact: ArtifactRef) -> ModelInput:
        return self._candidate(artifact).input

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        kept = self._candidate(artifact)
        graph = self._graph_of(kept.manifest)
        if graph is not None and self._graphs is not None:
            return self._graphs.predict(graph, self._content(graph), windows)
        measured = kept.manifest.measured
        if self._classical is not None and self._classical.reads(measured.format):
            try:
                return self._classical.answer(
                    measured,
                    self._content(measured),
                    windows,
                    channels=len(kept.input.channels),
                )
            except EvaluationContractError as error:
                raise UnreadableServedArtifactError(str(error)) from error
        raise UnservableArtifactError(self._unservable(artifact, kept.manifest))

    def embed(
        self, artifact: ArtifactRef, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        kept = self._candidate(artifact)
        graph = self._graph_of(kept.manifest)
        if graph is None:
            if self._classical is not None and self._classical.reads(kept.manifest.measured.format):
                raise EmbeddingUnavailableError(
                    f"the candidate under {artifact.key!r} is kept as "
                    f"{kept.manifest.measured.format!r}, a form with no representation to hand out"
                )
            raise UnservableArtifactError(self._unservable(artifact, kept.manifest))
        if self._graphs is None:
            raise UnservableArtifactError(self._unservable(artifact, kept.manifest))
        return self._graphs.embed(graph, self._content(graph), windows)

    def _graph_of(self, manifest: KeptCandidateManifest) -> KeptRepresentation | None:
        return manifest.find(OnnxGraphInference.FORMAT)

    def _candidate(self, artifact: ArtifactRef) -> _KeptCandidate:
        """The manifest under ``artifact`` and what the candidate takes, read once per process.

        Raises:
            ArtifactUnavailableError: If the manifest or the corpus manifest it names is not in
                the store.
            UnreadableServedArtifactError: If either is not a manifest this reads.
        """
        checksum = artifact.checksum
        if checksum not in self._candidates:
            try:
                manifest = self._manifests.decode(self._fetched(artifact))
                corpus = self._corpora.decode(self._fetched(manifest.corpus_manifest))
                model_input = ModelInput(
                    corpus=corpus.corpus,
                    window_length=corpus.window_length,
                    channels=corpus.channels,
                )
            except (EvaluationContractError, CatalogContractError, InvalidModelInputError) as error:
                raise UnreadableServedArtifactError(
                    f"the artifact under {artifact.key!r} is not a kept candidate this reads: "
                    f"{error}"
                ) from error
            self._candidates[checksum] = _KeptCandidate(manifest, model_input)
        return self._candidates[checksum]

    def _content(self, form: KeptRepresentation) -> bytes:
        checksum = form.artifact.checksum
        if checksum not in self._contents:
            self._contents[checksum] = self._fetched(form.artifact)
        return self._contents[checksum]

    def _fetched(self, artifact: ArtifactRef) -> bytes:
        """The bytes under ``artifact``, verified by the store.

        Raises:
            ArtifactUnavailableError: If the store has no such bytes, or not the bytes named.
        """
        try:
            return self._store.get(artifact)
        except ArtifactStoreError as error:
            raise ArtifactUnavailableError(
                f"artifact {artifact.key!r} is not in the store: {error}"
            ) from error

    def _unservable(self, artifact: ArtifactRef, manifest: KeptCandidateManifest) -> str:
        forms = ", ".join(form.format for form in manifest.representations)
        readable = [OnnxGraphInference.FORMAT] if self._graphs is not None else []
        if self._classical is not None:
            readable.append("the classical forms")
        return (
            f"the candidate under {artifact.key!r} is kept as {forms}, none of which this "
            f"process runs ({', '.join(readable) or 'nothing composed in'})"
        )
