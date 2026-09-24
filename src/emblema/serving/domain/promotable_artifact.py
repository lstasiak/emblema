from dataclasses import dataclass

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore
from emblema.serving.domain.exceptions import InvalidPromotableArtifactError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class PromotableArtifact:
    """An artifact a finished campaign kept, as Serving knows it: the only thing it may promote.

    A projection of what Evaluation announced, not a reference into its tables. It is built
    from the message a campaign publishes when it finishes, so the promise that a served model
    was measured holds without a foreign key across schemas and without asking the other
    context anything when a model is promoted. One exists only for a competitor whose artifact
    the campaign kept; the others have nothing to serve.

    Every kind of candidate is promotable on the same terms, which is why the kind is carried and
    never consulted here: it says which runtime loads the artifact, not whether it may be loaded.
    The standing and the scores are recorded for whoever chooses, and are not a condition either,
    for the reason ``CampaignScore`` gives.

    Invariants: no operating point — a metric at a budget — is scored twice.

    Attributes:
        origin: The campaign, task and competitor the artifact was measured as.
        kind: What the candidate is made of, which decides what can run the artifact.
        artifact: The fitted candidate, by reference and checksum.
        standing: Where the candidate ended against the campaign's control.
        scores: What it scored, one figure per operating point.
        completed_at: When the campaign finished, which is when the artifact became promotable.
    """

    origin: ArtifactOrigin
    kind: CandidateKind
    artifact: ArtifactRef
    standing: CandidateStanding
    scores: tuple[CampaignScore, ...]
    completed_at: UtcDateTime

    def __post_init__(self) -> None:
        points = [(score.metric, score.budget) for score in self.scores]
        if len(set(points)) != len(points):
            raise InvalidPromotableArtifactError(
                f"{self.origin.candidate} in campaign {self.origin.campaign} is scored twice "
                "at one operating point"
            )
