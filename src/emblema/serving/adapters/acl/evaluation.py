from typing import ClassVar

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.evaluated_candidate import EvaluatedCandidate
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.serving.application.use_cases.record_promotable_artifacts import (
    RecordPromotableArtifacts,
    RecordPromotableArtifactsCommand,
)
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.event_subscriber import EventSubscriber


class EvaluationAntiCorruptionLayer:
    """Where Serving hears what Evaluation announces, translated into Serving's own model.

    The one message it listens to is a campaign finishing, and it keeps from it only what can be
    served: a competitor the campaign kept no artifact of is dropped here rather than recorded
    as a promotable thing with nothing behind it, and so is one of a kind nothing here serves,
    such as a detector, whose artifact no served model could run. The verdict sentence is
    dropped too, since it is written for a person reading the campaign and nothing here acts
    on it.

    Attributes:
        SERVED_KINDS: The kinds of candidate a served model can be made of.

    It is the only module of this context that reads Evaluation's messages, so a change to them
    is absorbed here and reaches no other part of Serving.
    """

    SERVED_KINDS: ClassVar[frozenset[CandidateKind]] = frozenset(
        {CandidateKind.NEURAL, CandidateKind.CLASSICAL}
    )

    def __init__(self, record: RecordPromotableArtifacts) -> None:
        self._record = record

    def subscribe(self, subscriptions: EventSubscriber) -> None:
        """Listen to what Evaluation announces through ``subscriptions``."""
        subscriptions.subscribe(CampaignCompleted, self.on_campaign_completed)

    def on_campaign_completed(self, event: CampaignCompleted) -> None:
        self._record(
            RecordPromotableArtifactsCommand(
                artifacts=tuple(
                    self._promotable(event, candidate, artifact)
                    for candidate in event.candidates
                    if candidate.kind in self.SERVED_KINDS
                    and (artifact := candidate.artifact) is not None
                )
            )
        )

    @staticmethod
    def _promotable(
        event: CampaignCompleted, candidate: EvaluatedCandidate, artifact: ArtifactRef
    ) -> PromotableArtifact:
        return PromotableArtifact(
            origin=ArtifactOrigin(
                campaign=event.campaign, task=event.task, candidate=candidate.candidate
            ),
            kind=candidate.kind,
            artifact=artifact,
            standing=candidate.standing,
            scores=tuple(
                CampaignScore(
                    metric=metric.metric,
                    budget=metric.budget,
                    value=metric.value,
                    repeats=metric.repeats,
                )
                for metric in candidate.metrics
            ),
            completed_at=event.occurred_at,
        )
