from dataclasses import dataclass

from emblema.evaluation.application.use_cases.fulfil_campaign_order import FulfilCampaignOrder


@dataclass(frozen=True)
class Services:
    """The one use case a run of an order carries out."""

    fulfil_campaign_order: FulfilCampaignOrder
