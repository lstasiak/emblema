from dataclasses import dataclass, replace
from typing import Self

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.exceptions import InvalidServedModelError, ServedModelWithdrawnError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class ServedModel:
    """An artifact promoted to answer requests: what this context serves.

    Nothing is served that a finished campaign did not measure, and the model holds that by how
    it comes about: a new one is made by ``promoted`` out of a ``PromotableArtifact``, and the
    projection exists only for what a campaign announced it kept. The constructor is how a
    repository rebuilds a stored model, not a second way to make one.

    Withdrawal is the one transition and it is final. A model taken out of service is not put
    back; promoting the same artifact again makes a new model, so that each period of service is
    one record with its own beginning and end.

    Invariants: a model is not promoted before its campaign finished, and not withdrawn before
    it was promoted.

    Attributes:
        served_model_id: Identity of the model.
        origin: The campaign, task and competitor its artifact was measured as.
        kind: What the candidate is made of, which decides what can run the artifact.
        artifact: The artifact served, by reference and checksum.
        promoted_at: When it was promoted.
        withdrawn_at: When it was withdrawn; ``None`` while it serves.
    """

    served_model_id: ServedModelId
    origin: ArtifactOrigin
    kind: CandidateKind
    artifact: ArtifactRef
    promoted_at: UtcDateTime
    withdrawn_at: UtcDateTime | None = None

    def __post_init__(self) -> None:
        if self.withdrawn_at is not None and self.withdrawn_at < self.promoted_at:
            raise InvalidServedModelError(
                f"model {self.served_model_id} would be withdrawn before it was promoted"
            )

    @classmethod
    def promoted(
        cls, kept: PromotableArtifact, *, served_model_id: ServedModelId, at: UtcDateTime
    ) -> Self:
        """A model serving what a finished campaign kept, from ``at`` on.

        Raises:
            InvalidServedModelError: If ``at`` precedes the moment the campaign finished.
        """
        if at < kept.completed_at:
            raise InvalidServedModelError(
                f"campaign {kept.origin.campaign} had not finished at {at.value.isoformat()}"
            )
        return cls(
            served_model_id=served_model_id,
            origin=kept.origin,
            kind=kept.kind,
            artifact=kept.artifact,
            promoted_at=at,
        )

    @property
    def state(self) -> ServedModelState:
        if self.withdrawn_at is None:
            return ServedModelState.SERVING
        return ServedModelState.WITHDRAWN

    def withdraw(self, at: UtcDateTime) -> Self:
        """The model taken out of service at ``at``.

        Raises:
            ServedModelWithdrawnError: If it was already withdrawn.
            InvalidServedModelError: If ``at`` precedes its promotion.
        """
        if self.withdrawn_at is not None:
            raise ServedModelWithdrawnError(
                f"model {self.served_model_id} has already been withdrawn"
            )
        return replace(self, withdrawn_at=at)
