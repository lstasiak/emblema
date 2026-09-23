from dataclasses import dataclass

from emblema.evaluation.application.use_cases.complete_campaign import (
    CompleteCampaign,
    CompleteCampaignCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import UnknownCampaignCellError
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository


@dataclass(frozen=True, kw_only=True)
class RunCampaignCellCommand:
    """One point of a campaign's grid, as a worker receives it.

    Attributes:
        campaign: Which campaign the cell belongs to.
        cell: Which candidate, at which budget, under which seed.
    """

    campaign: CampaignId
    cell: CampaignCell


class RunCampaignCell:
    """Runs one cell, records it against the campaign, and closes the campaign if it was the last.

    A cell already recorded is returned rather than run again. That is what makes the work safe
    to hand to a queue: a broker that delivers a message twice, or a caller that resubmits a
    grid after a crash without knowing what survived, costs a lookup instead of a run — and,
    more to the point, cannot produce a second answer for a point the campaign has already
    recorded one for.
    """

    def __init__(
        self,
        campaigns: EvaluationCampaignRepository,
        candidates: CandidateProvider,
        complete_campaign: CompleteCampaign,
    ) -> None:
        self._campaigns = campaigns
        self._candidates = candidates
        self._complete = complete_campaign

    def __call__(self, command: RunCampaignCellCommand) -> CellResult:
        """Run the cell unless it has run, record it, and close the campaign if it is now whole.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            UnknownCampaignCellError: If the cell is not one of the campaign's grid.
            UnknownCandidateError: If the provider supplies no candidate of that name.
            FrozenTestSplitClosedError: If a run that is not the final one asked for the frozen
                side.
        """
        campaign = self._campaigns.get(command.campaign)
        recorded = self._recorded(campaign.results, command.cell)
        if recorded is not None:
            return recorded
        if command.cell not in campaign.pending():
            raise UnknownCampaignCellError(
                f"{command.cell} is not a cell of campaign {command.campaign}"
            )
        result = self._candidates.evaluate(
            CandidateEvaluation(
                task=campaign.task,
                cell=command.cell,
                purpose=campaign.purpose,
                retain=campaign.design.retains(command.cell),
                starts_from=campaign.design.get_candidate(command.cell.candidate).starts_from,
            )
        )
        campaign = campaign.record(result)
        self._campaigns.save(campaign)
        if campaign.is_complete:
            self._complete(CompleteCampaignCommand(campaign=campaign.campaign_id))
        return result

    @staticmethod
    def _recorded(results: tuple[CellResult, ...], cell: CampaignCell) -> CellResult | None:
        for result in results:
            if result.cell == cell:
                return result
        return None
