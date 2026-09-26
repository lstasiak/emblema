from typing import Protocol

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict


class VerdictMemo(Protocol):
    """Keeps verdicts already read, so that reading one again costs nothing.

    A verdict resamples every pairing of a campaign thousands of times, and a campaign at one
    revision always concludes the same, so what was read is kept under the campaign and the
    revision it was read at. What is kept, and for how long, is the adapter's business: a memo
    may forget, and a forgotten verdict is read again.
    """

    def recall(self, campaign: CampaignId, revision: int) -> CampaignVerdict | None:
        """The verdict kept for ``campaign`` at ``revision``, or ``None`` where none is."""
        ...

    def keep(self, campaign: CampaignId, revision: int, verdict: CampaignVerdict) -> None:
        """Keep ``verdict`` as what ``campaign`` concludes at ``revision``."""
        ...
