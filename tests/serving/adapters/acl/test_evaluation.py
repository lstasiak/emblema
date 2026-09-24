"""What Evaluation announces, heard and translated by Serving, down to a model in service.

The message is assembled by Evaluation's own assembler from a campaign closed by its own
aggregate, not written by hand: a hand-built message would agree with the translation by
construction, and the point is that the two contexts agree through the published language.
"""

import pytest

from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.serving.adapters.acl.evaluation import EvaluationAntiCorruptionLayer
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
from emblema.serving.application.use_cases.record_promotable_artifacts import (
    RecordPromotableArtifacts,
    RecordPromotableArtifactsCommand,
)
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.events.domain_event import EventId
from tests.evaluation.support import CAMPAIGN, CLOSED_AT, CONTENDER, CONTROL, TASK, closed_campaign
from tests.serving.support import FITTED, PROMOTED


class WatchedRecord(RecordPromotableArtifacts):
    """The use case the layer drives, remembering every command it was driven with."""

    def __init__(self, promotables: InMemoryPromotableArtifactRepository) -> None:
        super().__init__(promotables)
        self.commands: list[RecordPromotableArtifactsCommand] = []

    def __call__(self, command: RecordPromotableArtifactsCommand) -> None:
        self.commands.append(command)
        super().__call__(command)


class Heard:
    """Serving listening to one in-process publisher, with the fitted candidate in the store."""

    def __init__(self) -> None:
        self.store = InMemoryArtifactStore()
        self.kept = self.store.put(FITTED)
        self.promotables = InMemoryPromotableArtifactRepository()
        subscriptions = InMemoryEventSubscriber()
        self.record = WatchedRecord(self.promotables)
        EvaluationAntiCorruptionLayer(self.record).subscribe(subscriptions)
        self.events = InMemoryEventPublisher(subscriptions)

    def announced(self) -> CampaignCompleted:
        campaign = closed_campaign(self.kept)
        event = CampaignCompletedAssembler().assemble(
            campaign,
            campaign.verdict(),
            event_id=SequentialIdGenerator().generate(EventId),
            occurred_at=CLOSED_AT,
        )
        self.events.publish(event)
        return event


@pytest.fixture
def heard() -> Heard:
    return Heard()


def test_what_a_finished_campaign_kept_becomes_promotable_as_it_was_announced(
    heard: Heard,
) -> None:
    event = heard.announced()

    (recorded,) = heard.promotables.find_by_checksum(heard.kept.checksum)
    (announced,) = [entry for entry in event.candidates if entry.candidate == CONTENDER]
    assert recorded.origin == ArtifactOrigin(campaign=CAMPAIGN, task=TASK, candidate=CONTENDER)
    assert (recorded.kind, recorded.artifact, recorded.standing, recorded.completed_at) == (
        announced.kind,
        heard.kept,
        CandidateStanding.ESTABLISHED,
        CLOSED_AT,
    )
    assert recorded.scores == tuple(
        CampaignScore(metric=m.metric, budget=m.budget, value=m.value, repeats=m.repeats)
        for m in announced.metrics
    )


def test_a_competitor_the_campaign_kept_nothing_of_is_not_recorded(heard: Heard) -> None:
    event = heard.announced()

    (control,) = [entry for entry in event.candidates if entry.candidate == CONTROL]
    assert control.artifact is None
    (command,) = heard.record.commands
    assert [artifact.origin.candidate for artifact in command.artifacts] == [CONTENDER]


def test_an_announcement_heard_twice_is_recorded_once(heard: Heard) -> None:
    heard.announced()
    heard.announced()

    assert len(heard.promotables.find_by_checksum(heard.kept.checksum)) == 1


def test_what_a_finished_campaign_kept_can_be_put_into_service(heard: Heard) -> None:
    heard.announced()
    served = InMemoryServedModelRepository()
    promote = PromoteArtifact(
        heard.promotables, served, heard.store, SequentialIdGenerator(), FixedClock(PROMOTED)
    )

    model = served.get(promote(PromoteArtifactCommand(checksum=heard.kept.checksum)))

    assert model.origin.candidate == CONTENDER
    assert model.artifact == heard.kept
