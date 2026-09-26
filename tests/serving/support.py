"""Values the Serving tests build their cases from, each overridable where a test is about it."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.static_value import StaticValue
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.serving.adapters.in_memory.inference_runtime import StatedCandidate
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.inference_limits import InferenceLimits
from emblema.serving.domain.model_input import ModelInput
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


CORPUS = "test-corpus"
WINDOW_LENGTH = 10.0
# A vocabulary with every kind of channel a request can name: two measured and fitted, one
# static and fitted, and one the training data never observed, which has no statistics.
CHANNELS = (
    PublishedChannel(
        channel_id=1,
        corpus=CORPUS,
        channel="temperature",
        unit="K",
        statistics=PublishedChannelStatistics(3, 0.5, 1.0),
    ),
    PublishedChannel(
        channel_id=2,
        corpus=CORPUS,
        channel="pressure",
        unit="Pa",
        statistics=PublishedChannelStatistics(2, 0.25, 0.5),
    ),
    PublishedChannel(
        channel_id=3,
        corpus=CORPUS,
        channel="age",
        timeless=True,
        statistics=PublishedChannelStatistics(2, 60.0, 5.0),
    ),
    PublishedChannel(channel_id=4, corpus=CORPUS, channel="silent"),
)


def model_input(**overrides: Any) -> ModelInput:
    stated = ModelInput(corpus=CORPUS, window_length=WINDOW_LENGTH, channels=CHANNELS)
    return replace(stated, **overrides)


def observed_window(**overrides: Any) -> ObservedWindow:
    """Ten units of time with three readings on the fitted channels and one static value."""
    stated = ObservedWindow(
        start=0.0,
        length=WINDOW_LENGTH,
        observations=(
            ObservedValue(channel="temperature", time=1.0, value=0.5),
            ObservedValue(channel="pressure", time=2.0, value=0.75),
            ObservedValue(channel="temperature", time=8.0, value=1.5),
        ),
        static_features=(StaticValue(channel="age", value=65.0),),
    )
    return replace(stated, **overrides)


def stated(
    prediction: float = 42.0, embedding: tuple[float, ...] | None = (0.1, 0.2)
) -> StatedCandidate:
    return StatedCandidate(input=model_input(), prediction=prediction, embedding=embedding)


def limits(**overrides: Any) -> InferenceLimits:
    stated = InferenceLimits(max_windows_per_request=4, max_tokens_per_window=16)
    return replace(stated, **overrides)
