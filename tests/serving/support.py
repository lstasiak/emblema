"""Values the Serving tests build their cases from, each overridable where a test is about it."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.serving.domain.served_model import ServedModel
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.timestamps import UtcDateTime

CAMPAIGN = CampaignId(UUID(int=11))
OTHER_CAMPAIGN = CampaignId(UUID(int=12))
TASK = TaskId(UUID(int=13))
MODEL = ServedModelId(UUID(int=14))
TREES = CandidateRef("boosted_trees_per_channel")
FINISHED = UtcDateTime(datetime(2026, 9, 1, tzinfo=UTC))
PROMOTED = UtcDateTime(datetime(2026, 9, 2, tzinfo=UTC))
WITHDRAWN = UtcDateTime(datetime(2026, 9, 3, tzinfo=UTC))
FITTED = b"fitted candidate"
KEPT = ArtifactRef(key="durable/fitted", checksum=Checksum.of_bytes(FITTED))


def origin(**overrides: Any) -> ArtifactOrigin:
    return replace(ArtifactOrigin(campaign=CAMPAIGN, task=TASK, candidate=TREES), **overrides)


def score(**overrides: Any) -> CampaignScore:
    return replace(CampaignScore(metric="rmse", budget=200, value=17.5, repeats=5), **overrides)


def promotable(**overrides: Any) -> PromotableArtifact:
    stated = PromotableArtifact(
        origin=origin(),
        kind=CandidateKind.CLASSICAL,
        artifact=KEPT,
        standing=CandidateStanding.ESTABLISHED,
        scores=(score(budget=50, value=24.0), score(), score(budget=None, value=15.0)),
        completed_at=FINISHED,
    )
    return replace(stated, **overrides)


def served(**overrides: Any) -> ServedModel:
    stated = ServedModel.promoted(promotable(), served_model_id=MODEL, at=PROMOTED)
    return replace(stated, **overrides)
