from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class CampaignPosition:
    """Where a campaign stands in the order its list is read in: design time, then identity.

    A page continues after a position rather than after a row, so the next page needs nothing
    stored: a campaign designed between two pages leaves the position where it was.

    Attributes:
        opened_at: When the campaign was designed; later ones are listed first.
        campaign_id: Identity, which orders campaigns designed at the same instant.
    """

    opened_at: UtcDateTime
    campaign_id: CampaignId
