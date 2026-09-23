from dataclasses import dataclass

from emblema.evaluation.application.use_cases.complete_campaign import (
    CompleteCampaign,
    CompleteCampaignCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    UnknownCampaignCellError,
)
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

    Losing the race to record is not losing the work. Another process finishing a cell of the
    same grid while this one was running its own is the ordinary case once a campaign is worked
    by more than one worker, so the campaign is read again and the result put against the state
    that now stands. What was expensive has already happened, and the cell is answered once.
    """

    # A retry is lost only to another process recording a cell, and a grid holds finitely many.
    # The bound is here so that a store that keeps refusing surfaces as a failure rather than
    # as a worker that spins.
    ATTEMPTS = 8

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
        declared = campaign.design.get_candidate(command.cell.candidate)
        result = self._candidates.evaluate(
            CandidateEvaluation(
                task=campaign.task,
                cell=command.cell,
                purpose=campaign.purpose,
                retain=campaign.design.retains(command.cell),
                declared=declared,
            )
        )
        return self._record(campaign, command, result)

    def _record(
        self, campaign: EvaluationCampaign, command: RunCampaignCellCommand, result: CellResult
    ) -> CellResult:
        """Put ``result`` against the campaign, reading it again while others get there first.

        Whoever records the cell that completes the grid is whoever reads a state the rest of
        the grid is already in, which is why closing it is decided on the state that was
        actually written and not on the one this process started from.

        Raises:
            CampaignChangedElsewhereError: If the campaign kept moving for as many attempts as
                are allowed.
        """
        for _ in range(self.ATTEMPTS):
            already = self._recorded(campaign.results, command.cell)
            if already is not None:
                return already
            recorded = campaign.record(result)
            try:
                self._campaigns.save(recorded, seen=campaign.revision)
            except CampaignChangedElsewhereError:
                campaign = self._campaigns.get(command.campaign)
                continue
            if recorded.is_complete:
                self._complete(CompleteCampaignCommand(campaign=recorded.campaign_id))
            return result
        raise CampaignChangedElsewhereError(
            f"campaign {command.campaign} moved on under every one of {self.ATTEMPTS} attempts "
            f"to record {command.cell}"
        )

    @staticmethod
    def _recorded(results: tuple[CellResult, ...], cell: CampaignCell) -> CellResult | None:
        for result in results:
            if result.cell == cell:
                return result
        return None
