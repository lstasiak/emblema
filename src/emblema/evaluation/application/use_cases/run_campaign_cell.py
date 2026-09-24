from dataclasses import dataclass

from emblema.evaluation.application.use_cases.record_cell_result import (
    RecordCellResult,
    RecordCellResultCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
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
    """Runs one cell and records it against the campaign, which closes it if it was the last.

    A cell already recorded is returned rather than run again. That is what makes the work safe
    to hand to a queue: a broker that delivers a message twice, or a caller that resubmits a
    grid after a crash without knowing what survived, costs a lookup instead of a run — and,
    more to the point, cannot produce a second answer for a point the campaign has already
    recorded one for.

    Running is the candidate's and recording is ``RecordCellResult``'s; this is the two in one
    process. A cell run on a machine that cannot reach the registry goes through the same two
    halves with an order between them.
    """

    def __init__(
        self,
        campaigns: EvaluationCampaignRepository,
        candidates: CandidateProvider,
        record_cell_result: RecordCellResult,
    ) -> None:
        self._campaigns = campaigns
        self._candidates = candidates
        self._record = record_cell_result

    def __call__(self, command: RunCampaignCellCommand) -> CellResult:
        """Run the cell unless it has run, record it, and close the campaign if it is now whole.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            UnknownCampaignCellError: If the cell is not one of the campaign's grid.
            UnknownCandidateError: If the provider supplies no candidate of that name.
            CandidateMismatchError: If the campaign recorded the candidate as something other
                than what the provider supplies.
            FrozenTestSplitClosedError: If a run that is not the final one asked for the frozen
                side.
            CampaignChangedElsewhereError: If the campaign kept moving for as many attempts as
                are allowed to record the result.
        """
        campaign = self._campaigns.get(command.campaign)
        recorded = RecordCellResult.recorded(campaign.results, command.cell)
        if recorded is not None:
            return recorded
        if command.cell not in campaign.pending():
            raise UnknownCampaignCellError(
                f"{command.cell} is not a cell of campaign {command.campaign}"
            )
        result = self._candidates.evaluate(campaign.evaluation_of(command.cell))
        return self._record(RecordCellResultCommand(campaign=command.campaign, result=result))
