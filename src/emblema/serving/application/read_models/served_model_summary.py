from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class ServedModelSummary:
    """A served model as a list shows it: identity, state, provenance and the artifact served.

    A read model rather than the aggregate: flat, with the origin spelled out as the identities
    a client follows to the campaign and the task, and nothing a client cannot act on.

    Attributes:
        served_model_id: Identity of the model.
        state: Whether it answers requests.
        campaign: The finished campaign that measured its artifact.
        task: The task that campaign answered.
        candidate: What the campaign called the competitor.
        kind: What the candidate is made of.
        artifact: The artifact served, by reference and checksum.
        promoted_at: When it was promoted.
        withdrawn_at: When it was withdrawn; ``None`` while it serves.
    """

    served_model_id: ServedModelId
    state: ServedModelState
    campaign: CampaignId
    task: TaskId
    candidate: CandidateRef
    kind: CandidateKind
    artifact: ArtifactRef
    promoted_at: UtcDateTime
    withdrawn_at: UtcDateTime | None

    @classmethod
    def of(cls, model: ServedModel) -> Self:
        return cls(
            served_model_id=model.served_model_id,
            state=model.state,
            campaign=model.origin.campaign,
            task=model.origin.task,
            candidate=model.origin.candidate,
            kind=model.kind,
            artifact=model.artifact,
            promoted_at=model.promoted_at,
            withdrawn_at=model.withdrawn_at,
        )
