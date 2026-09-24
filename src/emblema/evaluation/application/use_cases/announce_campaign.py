from dataclasses import dataclass

from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.events.domain_event import EventId
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class AnnounceCampaignCommand:
    """Request to publish again what a closed campaign concluded.

    Attributes:
        campaign: Which campaign to announce.
    """

    campaign: CampaignId


class AnnounceCampaign:
    """Publishes a closed campaign's conclusion again, for a subscriber that missed it.

    Closing a campaign stores it and then publishes, in process. A subscriber that fails in
    between leaves the campaign closed and the message undelivered, and closing cannot be
    repeated, so without this the loss would be permanent. The message is assembled from the
    stored campaign exactly as closing assembled it, dated when the campaign closed rather than
    now, so a subscriber that records it by what it says receives the same thing twice and keeps
    it once. It is the manual form of what an outbox would do on its own.
    """

    def __init__(
        self,
        campaigns: EvaluationCampaignRepository,
        outcomes: CampaignCompletedAssembler,
        ids: IdGenerator,
        events: EventPublisher,
    ) -> None:
        self._campaigns = campaigns
        self._outcomes = outcomes
        self._ids = ids
        self._events = events

    def __call__(self, command: AnnounceCampaignCommand) -> CampaignCompleted:
        """Publish the campaign's conclusion and return the message.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            CampaignNotCompletedError: If it has not been closed, so has nothing to announce.
        """
        campaign = self._campaigns.get(command.campaign)
        closed_at = campaign.completion()
        announced = self._outcomes.assemble(
            campaign,
            campaign.verdict(),
            event_id=self._ids.generate(EventId),
            occurred_at=closed_at,
        )
        self._events.publish(announced)
        return announced
