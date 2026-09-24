from __future__ import annotations

import json
from typing import Final

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.exceptions import MalformedKeptCandidateManifestError
from emblema.evaluation.contracts.kept_candidate_manifest import KeptCandidateManifest
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

type _Document = dict[str, object]


class KeptCandidateManifestJson:
    """The manifest of a kept candidate as JSON, the form it is stored in.

    Compact and key-ordered, so one candidate kept twice is one manifest in a content-addressed
    store. Part of the published language because another context reads it from the artifact
    store to find the form it can run, with nothing but the standard library.
    """

    FORMAT: Final = "emblema.kept-candidate-manifest"
    # Bumped when the meaning of a field changes; a new field is one an older reader ignores.
    VERSION: Final = 1

    def encode(self, manifest: KeptCandidateManifest) -> bytes:
        document: _Document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "kind": str(manifest.kind),
            "corpus_manifest": self._ref_document(manifest.corpus_manifest),
            "measured_as": manifest.measured_as,
            "representations": [
                {
                    "format": representation.format,
                    "artifact": self._ref_document(representation.artifact),
                    "deviation": representation.deviation,
                }
                for representation in manifest.representations
            ],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> KeptCandidateManifest:
        """The manifest those bytes describe.

        Raises:
            MalformedKeptCandidateManifestError: If the bytes are not a manifest of a version
                this reads, a field is missing or of the wrong type, or the manifest breaks its
                own rules.
        """
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MalformedKeptCandidateManifestError(f"manifest is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise MalformedKeptCandidateManifestError("manifest is not an object")
        if document.get("format") != self.FORMAT or document.get("version") != self.VERSION:
            raise MalformedKeptCandidateManifestError(
                f"manifest is a {document.get('format')!r} of version {document.get('version')!r}"
            )
        try:
            return self._manifest(_Fields(document))
        except MalformedKeptCandidateManifestError:
            raise
        except ValueError as error:
            raise MalformedKeptCandidateManifestError(
                f"manifest is not well formed: {error}"
            ) from error

    def _manifest(self, fields: _Fields) -> KeptCandidateManifest:
        return KeptCandidateManifest(
            kind=CandidateKind(fields.text("kind")),
            corpus_manifest=self._ref(fields.fields("corpus_manifest")),
            representations=tuple(
                KeptRepresentation(
                    format=item.text("format"),
                    artifact=self._ref(item.fields("artifact")),
                    deviation=item.optional_number("deviation"),
                )
                for item in fields.each("representations")
            ),
            measured_as=fields.text("measured_as"),
        )

    @staticmethod
    def _ref_document(ref: ArtifactRef) -> _Document:
        return {
            "key": ref.key,
            "checksum": {"algorithm": str(ref.checksum.algorithm), "digest": ref.checksum.digest},
        }

    @staticmethod
    def _ref(fields: _Fields) -> ArtifactRef:
        checksum = fields.fields("checksum")
        return ArtifactRef(
            fields.text("key"),
            Checksum(HashAlgorithm(checksum.text("algorithm")), checksum.text("digest")),
        )


class _Fields:
    """A JSON object read one field at a time; a failure names the field and what it had to be."""

    def __init__(self, document: _Document) -> None:
        self._document = document

    def text(self, key: str) -> str:
        value = self._required(key)
        if not isinstance(value, str):
            raise MalformedKeptCandidateManifestError(f"field {key!r} must be text, got {value!r}")
        return value

    def optional_number(self, key: str) -> float | None:
        value = self._required(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MalformedKeptCandidateManifestError(
                f"field {key!r} must be a number or null, got {value!r}"
            )
        return float(value)

    def fields(self, key: str) -> _Fields:
        value = self._required(key)
        if not isinstance(value, dict):
            raise MalformedKeptCandidateManifestError(
                f"field {key!r} must be an object, got {value!r}"
            )
        return _Fields(value)

    def each(self, key: str) -> list[_Fields]:
        value = self._required(key)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise MalformedKeptCandidateManifestError(f"field {key!r} must be a list of objects")
        return [_Fields(item) for item in value]

    def _required(self, key: str) -> object:
        if key not in self._document:
            raise MalformedKeptCandidateManifestError(f"field {key!r} is missing")
        return self._document[key]
