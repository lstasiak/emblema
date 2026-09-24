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

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        """Store the campaign, provided nothing has changed it since revision ``seen``.

        A grid is worked by more than one process, and each of them reads the whole campaign,
        runs one cell and writes the whole campaign back. Writing over a state one never read
        would drop the cells another process recorded in between, so what a caller may write
        over is stated rather than assumed: the revision it read.

        Args:
            campaign: The campaign as the caller would now have it.
            seen: The revision the caller read, which is the state it is entitled to replace.
                A campaign nobody has stored yet is written whatever this says.

        Raises:
            CampaignChangedElsewhereError: If it has moved on since that revision.
        """
        ...
