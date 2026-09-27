from dataclasses import dataclass

from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary
from emblema.evaluation.application.read_models.campaign_view import CampaignView
from emblema.evaluation.application.read_models.candidate_curve import CandidateCurve
from emblema.evaluation.application.read_models.verdict_view import VerdictView
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.evaluation.ports.verdict_memo import VerdictMemo


@dataclass(frozen=True, kw_only=True)
class ViewCampaignQuery:
    """Which campaign to show.

    Attributes:
        campaign: Identity of the campaign.
    """

    campaign: CampaignId


class ViewCampaign:
    """Shows one campaign: its design, every candidate's curve so far, and the verdict if any.

    The verdict is read by the rules the campaign registered, which resamples every pairing
    thousands of times and takes the better part of a second; a campaign at one revision always
    concludes the same, so a verdict read is kept in a memo under the campaign's revision, and a
    campaign still gaining cells has none to keep.
    """

    def __init__(self, campaigns: EvaluationCampaignRepository, verdicts: VerdictMemo) -> None:
        self._campaigns = campaigns
        self._verdicts = verdicts

    def __call__(self, query: ViewCampaignQuery) -> CampaignView:
        """The campaign in full.

        Raises:
            CampaignNotFoundError: If no campaign is stored under that identity.
        """
        campaign = self._campaigns.get(query.campaign)
        verdict = self._verdict_of(campaign)
        return CampaignView(
            summary=CampaignSummary.of(CampaignOverview.of(campaign)),
            curves=tuple(
                CandidateCurve.of(campaign, candidate) for candidate in campaign.design.candidates
            ),
            verdict=None if verdict is None else VerdictView.of(verdict),
        )

    def _verdict_of(self, campaign: EvaluationCampaign) -> CampaignVerdict | None:
        if not campaign.is_finished or campaign.selects:
            return None
        kept = self._verdicts.recall(campaign.campaign_id, campaign.revision)
        if kept is not None:
            return kept
        verdict = campaign.verdict()
        self._verdicts.keep(campaign.campaign_id, campaign.revision, verdict)
        return verdict
