from collections.abc import Sequence

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.kept_candidate_manifest import KeptCandidateManifest
from emblema.evaluation.contracts.kept_candidate_manifest_json import KeptCandidateManifestJson
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class KeptCandidates:
    """Where a runtime puts what it fitted, in every form, under one manifest naming the forms.

    Every runtime keeps a candidate the same way whatever it fitted: each form goes into the
    store as an artifact of its own, then the manifest naming them all goes in after, and the
    manifest's reference is what the run reports. The order is forced by a content-addressed
    store, where a manifest can only name bytes that already exist; stating it once here keeps
    the runtimes from each stating it slightly differently.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._json = KeptCandidateManifestJson()

    def keep(
        self,
        kind: CandidateKind,
        *,
        corpus_manifest: ArtifactRef,
        measured: RepresentationBytes,
        derived: Sequence[RepresentationBytes] = (),
    ) -> ArtifactRef:
        """Store ``measured`` and every ``derived`` form, then the manifest that names them.

        Args:
            kind: What the candidate is made of.
            corpus_manifest: The published manifest of the corpus the candidate was fitted to.
            measured: The form whose answers the campaign scored.
            derived: The forms made from it for another context to run, each with the deviation
                its answers showed from the measured form's.

        Returns:
            Reference to the manifest, which is what the campaign records and announces.

        Raises:
            InvalidKeptCandidateManifestError: If the forms do not make a manifest — a format
                listed twice, or a derived form without a deviation.
        """
        representations = tuple(
            KeptRepresentation(
                format=form.format,
                artifact=self._store.put(form.content),
                deviation=form.deviation,
            )
            for form in (measured, *derived)
        )
        manifest = KeptCandidateManifest(
            kind=kind,
            corpus_manifest=corpus_manifest,
            representations=representations,
            measured_as=measured.format,
        )
        return self._store.put(self._json.encode(manifest))

    def read(self, manifest: ArtifactRef) -> KeptCandidateManifest:
        """The manifest stored under ``manifest``.

        Raises:
            MalformedKeptCandidateManifestError: If the bytes are not a manifest this reads.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If the stored bytes do not hash to the reference's checksum.
        """
        return self._json.decode(self._store.get(manifest))
