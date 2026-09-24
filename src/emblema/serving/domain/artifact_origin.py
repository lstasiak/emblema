from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId


@dataclass(frozen=True, kw_only=True)
class ArtifactOrigin:
    """Where an artifact was measured: which campaign, over which task, as which competitor.

    Written in the identities Evaluation publishes rather than in words of this context's own,
    because the one thing a served model can truthfully say about its provenance is what the
    campaign that measured it said. A campaign names each competitor once, so an origin names at
    most one artifact; the converse does not hold, since two competitors may have been fitted to
    the same bytes, which is why a promotion may have to name both parts of the origin.

    Attributes:
        campaign: The finished campaign that kept the artifact.
        task: The task the campaign answered.
        candidate: What the campaign called the competitor the artifact is a fit of.
    """

    campaign: CampaignId
    task: TaskId
    candidate: CandidateRef
