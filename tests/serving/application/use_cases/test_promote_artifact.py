import pytest

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.use_cases.promote_artifact import (
    PromoteArtifact,
    PromoteArtifactCommand,
)
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModel,
    WithdrawServedModelCommand,
)
from emblema.serving.domain.exceptions import (
    AmbiguousArtifactError,
    ArtifactAlreadyServedError,
    ArtifactNotPromotableError,
    ArtifactUnavailableError,
)
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.serving.support import (
    CAMPAIGN,
    FITTED,
    KEPT,
    OTHER_CAMPAIGN,
    PROMOTED,
    origin,
    promotable,
)

OTHER_TREES = CandidateRef("boosted_trees_pooled")


class Serving:
    """The context over adapters that touch nothing, with one artifact in the store."""

    def __init__(self) -> None:
        self.store = InMemoryArtifactStore()
        self.stored = self.store.put(FITTED)
        self.promotables = InMemoryPromotableArtifactRepository()
        self.served = InMemoryServedModelRepository()
        clock = FixedClock(PROMOTED)
        self.promote = PromoteArtifact(
            self.promotables, self.served, self.store, SequentialIdGenerator(), clock
        )
        self.withdraw = WithdrawServedModel(self.served, clock)

    def kept(self, **overrides: object) -> PromotableArtifact:
        artifact = promotable(artifact=self.stored, **overrides)
        self.promotables.save(artifact)
        return artifact


@pytest.fixture
def serving() -> Serving:
    return Serving()


def test_an_artifact_no_campaign_kept_is_refused(serving: Serving) -> None:
    with pytest.raises(ArtifactNotPromotableError, match="no finished campaign kept"):
        serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))


def test_an_artifact_is_refused_out_of_a_campaign_that_did_not_keep_it(serving: Serving) -> None:
    serving.kept()

    with pytest.raises(ArtifactNotPromotableError, match=str(OTHER_CAMPAIGN)):
        serving.promote(
            PromoteArtifactCommand(checksum=serving.stored.checksum, campaign=OTHER_CAMPAIGN)
        )


def test_an_artifact_several_campaigns_kept_is_refused_until_one_is_named(
    serving: Serving,
) -> None:
    serving.kept()
    serving.kept(origin=origin(campaign=OTHER_CAMPAIGN))

    with pytest.raises(AmbiguousArtifactError, match="name the campaign"):
        serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))
    model = serving.promote(
        PromoteArtifactCommand(checksum=serving.stored.checksum, campaign=OTHER_CAMPAIGN)
    )

    assert serving.served.get(model).origin.campaign == OTHER_CAMPAIGN


def test_an_artifact_one_campaign_kept_as_two_competitors_is_refused_until_one_is_named(
    serving: Serving,
) -> None:
    serving.kept()
    serving.kept(origin=origin(candidate=OTHER_TREES))

    with pytest.raises(AmbiguousArtifactError, match="where one campaign kept it twice"):
        serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum, campaign=CAMPAIGN))
    model = serving.promote(
        PromoteArtifactCommand(checksum=serving.stored.checksum, candidate=OTHER_TREES)
    )

    assert serving.served.get(model).origin.candidate == OTHER_TREES


def test_an_artifact_is_refused_as_a_competitor_that_was_not_fitted_to_it(
    serving: Serving,
) -> None:
    serving.kept()

    with pytest.raises(ArtifactNotPromotableError, match=f"as {OTHER_TREES} in campaign"):
        serving.promote(
            PromoteArtifactCommand(
                checksum=serving.stored.checksum, campaign=CAMPAIGN, candidate=OTHER_TREES
            )
        )


def test_an_artifact_whose_bytes_are_not_in_the_store_is_refused(serving: Serving) -> None:
    # Kept by a campaign, by a reference to content this store was never given.
    serving.promotables.save(promotable(artifact=KEPT))

    with pytest.raises(ArtifactUnavailableError, match="not in the store"):
        serving.promote(PromoteArtifactCommand(checksum=KEPT.checksum))


def test_an_artifact_already_in_service_is_not_promoted_twice(serving: Serving) -> None:
    serving.kept()
    serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))

    with pytest.raises(ArtifactAlreadyServedError):
        serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))


def test_a_withdrawn_artifact_is_promoted_again_as_a_new_model(serving: Serving) -> None:
    serving.kept()
    first = serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))
    serving.withdraw(WithdrawServedModelCommand(served_model=first))

    second = serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))

    assert second != first
    assert serving.served.get(first).state is ServedModelState.WITHDRAWN
    assert serving.served.get(second).state is ServedModelState.SERVING


@pytest.mark.parametrize("kind", list(CandidateKind))
def test_every_kind_of_candidate_is_promoted_on_the_same_terms(
    serving: Serving, kind: CandidateKind
) -> None:
    kept = serving.kept(kind=kind)

    model = serving.served.get(
        serving.promote(PromoteArtifactCommand(checksum=serving.stored.checksum))
    )

    assert (model.origin, model.kind, model.artifact) == (kept.origin, kind, serving.stored)
    assert model.promoted_at == PROMOTED
