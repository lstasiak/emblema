from dataclasses import dataclass

from emblema.evaluation.application.use_cases.complete_campaign import (
    CompleteCampaign,
    CompleteCampaignCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    UnknownCampaignCellError,
)
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository


@dataclass(frozen=True, kw_only=True)
class RecordCellResultCommand:
    """A cell's result, to be put against its campaign.

    Attributes:
        campaign: Which campaign the cell belongs to.
        result: What the cell produced.
    """

    campaign: CampaignId
    result: CellResult


class RecordCellResult:
    """Puts one cell's result against its campaign, and closes the campaign if it was the last.

    The one way a result reaches a campaign, whether the cell was run by a worker a moment ago
    or on another machine last night: a campaign is only ever written over by a process that
    read it, and whoever records the cell that completes the grid closes it.

    A cell already recorded keeps the result it has. Losing the race to record is not losing
    the work: another process finishing a cell of the same grid is the ordinary case once a
    campaign is worked by more than one, so the campaign is read again and the result put
    against the state that now stands.
    """

    # A retry is lost only to another process recording a cell, and a grid holds finitely many.
    # The bound is here so that a store that keeps refusing surfaces as a failure rather than
    # as a writer that spins.
    ATTEMPTS = 8

    def __init__(
        self, campaigns: EvaluationCampaignRepository, complete_campaign: CompleteCampaign
    ) -> None:
        self._campaigns = campaigns
        self._complete = complete_campaign

    def __call__(self, command: RecordCellResultCommand) -> CellResult:
        """Record the result unless the cell has one, and return the one that stands.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            UnknownCampaignCellError: If the grid does not hold the cell.
            CampaignChangedElsewhereError: If the campaign kept moving for as many attempts as
                are allowed.
        """
        cell = command.result.cell
        for _ in range(self.ATTEMPTS):
            campaign = self._campaigns.get(command.campaign)
            already = self.recorded(campaign.results, cell)
            if already is not None:
                return already
            if cell not in campaign.pending():
                raise UnknownCampaignCellError(
                    f"{cell} is not a cell of campaign {command.campaign}"
                )
            recorded = campaign.record(command.result)
            try:
                self._campaigns.save(recorded, seen=campaign.revision)
            except CampaignChangedElsewhereError:
                continue
            if recorded.is_complete:
                self._complete(CompleteCampaignCommand(campaign=recorded.campaign_id))
            return command.result
        raise CampaignChangedElsewhereError(
            f"campaign {command.campaign} moved on under every one of {self.ATTEMPTS} attempts "
            f"to record {cell}"
        )

    @staticmethod
    def recorded(results: tuple[CellResult, ...], cell: CampaignCell) -> CellResult | None:
        """The result already standing for ``cell``, if any."""
        for result in results:
            if result.cell == cell:
                return result
        return None
