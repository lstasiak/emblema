from dataclasses import dataclass

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.candidate_metric import CandidateMetric
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class EvaluatedCandidate:
    """One competitor as a finished campaign leaves it: what it is, what it scored, where it ended.

    The artifact is the candidate as it was fitted to the task, held by reference and checksum,
    and it is what another context may serve. It is absent where the campaign kept none — an arm
    that exists to be measured against rather than deployed, and every cell the campaign did not
    designate as its operating point.

    Attributes:
        candidate: What the campaign called this competitor.
        kind: What it is made of, which decides what can run it.
        artifact: The fitted candidate, where the campaign kept one.
        standing: Where it ended against the control, at the campaign's operating point.
        metrics: What it scored, one figure per operating point it was measured at.
    """

    candidate: CandidateRef
    kind: CandidateKind
    artifact: ArtifactRef | None
    standing: CandidateStanding
    metrics: tuple[CandidateMetric, ...]
