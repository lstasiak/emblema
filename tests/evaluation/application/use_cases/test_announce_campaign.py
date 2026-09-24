from dataclasses import replace

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.announce_campaign import (
    AnnounceCampaign,
    AnnounceCampaignCommand,
)
from emblema.evaluation.application.use_cases.complete_campaign import (
    CompleteCampaign,
    CompleteCampaignCommand,
)
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignNotCompletedError,
    CampaignNotFoundError,
    SelectionHasNoVerdictError,
)
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.evaluation.support import (
    CAMPAIGN,
    CLOSED_AT,
    SELECTED_BY,
    artifact,
    campaign,
    ran_campaign,
    selection,
)

KEPT = artifact("contender")


class Announcing:
    def __init__(self, stored: EvaluationCampaign | None) -> None:
        self.campaigns = InMemoryEvaluationCampaignRepository()
        if stored is not None:
            self.campaigns.save(stored, seen=0)
        subscriptions = InMemoryEventSubscriber()
        self.published: list[CampaignCompleted] = []
        subscriptions.subscribe(CampaignCompleted, self.published.append)
        self.events = InMemoryEventPublisher(subscriptions)
        self.ids = SequentialIdGenerator()
        self.announce = AnnounceCampaign(
            self.campaigns, CampaignCompletedAssembler(), self.ids, self.events
        )


def test_a_closed_campaign_is_announced_as_closing_announced_it() -> None:
    # Closed through the use case that publishes on closing, so the comparison is with what a
    # subscriber received the first time and not with a message built by hand.
    announcing = Announcing(ran_campaign(KEPT))
    closed = CompleteCampaign(
        announcing.campaigns,
        CampaignCompletedAssembler(),
        FixedClock(CLOSED_AT),
        announcing.ids,
        announcing.events,
    )(CompleteCampaignCommand(campaign=CAMPAIGN))

    again = announcing.announce(AnnounceCampaignCommand(campaign=CAMPAIGN))

    assert closed is not None, "a comparison announces what it concluded when it closes"
    assert replace(again, event_id=closed.event_id) == closed
    assert announcing.published == [closed, again]


def test_a_campaign_still_running_has_nothing_to_announce() -> None:
    announcing = Announcing(campaign())

    with pytest.raises(CampaignNotCompletedError):
        announcing.announce(AnnounceCampaignCommand(campaign=CAMPAIGN))

    assert announcing.published == []


def test_a_campaign_nobody_stored_is_refused() -> None:
    with pytest.raises(CampaignNotFoundError):
        Announcing(None).announce(AnnounceCampaignCommand(campaign=CAMPAIGN))


def test_a_selection_concludes_nothing_so_has_nothing_to_announce() -> None:
    announcing = Announcing(selection())

    with pytest.raises(SelectionHasNoVerdictError):
        announcing.announce(AnnounceCampaignCommand(campaign=SELECTED_BY))

    assert announcing.published == []
