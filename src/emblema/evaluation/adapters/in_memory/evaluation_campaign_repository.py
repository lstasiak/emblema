from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignNotFoundError,
)


class InMemoryEvaluationCampaignRepository:
    """Keeps campaigns in a dictionary, for a process that designs and runs one in a single go."""

    def __init__(self) -> None:
        self._campaigns: dict[CampaignId, EvaluationCampaign] = {}

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        try:
            return self._campaigns[campaign_id]
        except KeyError as error:
            raise CampaignNotFoundError(f"no campaign stored under {campaign_id}") from error

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        stored = self._campaigns.get(campaign.campaign_id)
        if stored is not None and stored.revision != seen:
            raise CampaignChangedElsewhereError(
                f"campaign {campaign.campaign_id} was read at revision {seen} and stands at "
                f"{stored.revision}"
            )
        self._campaigns[campaign.campaign_id] = campaign
