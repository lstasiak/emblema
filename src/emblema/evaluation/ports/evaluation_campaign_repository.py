from typing import Protocol

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign


class EvaluationCampaignRepository(Protocol):
    """Keeps campaigns, which is what makes a grid survive the process that runs it.

    A campaign's state is what has been run, and it is kept here rather than in the queue: a
    broker that lost its messages would cost a resubmission, while a grid whose progress lived
    in the messages would have to start again.
    """

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        """The campaign stored under that identity.

        Raises:
            CampaignNotFoundError: If no campaign is stored under it.
        """
        ...

    def save(self, campaign: EvaluationCampaign) -> None:
        """Store the campaign, replacing any earlier state of it."""
        ...
