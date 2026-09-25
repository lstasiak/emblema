from dataclasses import dataclass

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.exceptions import InvalidKeptCandidateManifestError
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class KeptCandidateManifest:
    """What a campaign kept of a candidate: every form it is stored in, and which was measured.

    The artifact a campaign announces for a kept competitor is this manifest, and the manifest
    names the bytes. That is what lets the announcement stay one reference while the candidate
    exists in more than one form — the state a runtime fitted, the graph another context runs —
    and lets a new form arrive without the announcement, the domain or the schema learning of
    it. It travels through the artifact store rather than through a call, so it is part of the
    published language and reads with nothing but the standard library.

    Invariants: at least one form; no format twice; the measured form is among the forms and
    carries no deviation, and every other form carries one.

    Attributes:
        kind: What the candidate is made of, which says what forms to expect.
        corpus_manifest: The published manifest of the corpus the candidate was fitted to, whose
            vocabulary its channel identifiers mean.
        representations: Every form the candidate is stored in.
        measured_as: Format of the form whose answers the campaign scored.
    """

    kind: CandidateKind
    corpus_manifest: ArtifactRef
    representations: tuple[KeptRepresentation, ...]
    measured_as: str

    def __post_init__(self) -> None:
        formats = [representation.format for representation in self.representations]
        if not formats:
            raise InvalidKeptCandidateManifestError("a kept candidate is stored in some form")
        if len(set(formats)) != len(formats):
            raise InvalidKeptCandidateManifestError(f"a format is listed twice among {formats}")
        if self.measured_as not in formats:
            raise InvalidKeptCandidateManifestError(
                f"the measured form {self.measured_as!r} is not among the forms {formats}"
            )
        for representation in self.representations:
            measured = representation.format == self.measured_as
            if measured != (representation.deviation is None):
                raise InvalidKeptCandidateManifestError(
                    f"the form {representation.format!r} "
                    + (
                        "was measured and reports a deviation"
                        if measured
                        else "was derived and reports no deviation"
                    )
                )

    @property
    def measured(self) -> KeptRepresentation:
        """The form whose answers the campaign scored."""
        return next(r for r in self.representations if r.format == self.measured_as)

    def find(self, format: str) -> KeptRepresentation | None:
        """The form stored as ``format``, or ``None`` where the candidate is not stored that way."""
        return next((r for r in self.representations if r.format == format), None)
