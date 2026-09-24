from dataclasses import dataclass

from emblema.evaluation.application.use_cases.fulfil_campaign_order import (
    FulfilCampaignOrderCommand,
)


@dataclass(frozen=True, kw_only=True)
class CampaignRunInvocation:
    """One run of an order, as the arguments and the code this machine runs settle it.

    Attributes:
        command: The order to run, the revision it is run at, and what to resume from.
    """

    command: FulfilCampaignOrderCommand
