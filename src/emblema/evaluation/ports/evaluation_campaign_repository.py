from typing import Protocol

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign


class EvaluationCampaignRepository(Protocol):
    """Keeps campaigns and what their cells produced, which is what makes a grid survive a process.

    A campaign's state is what has been run, and it is kept here rather than in the queue: a
    broker that lost its messages would cost a resubmission, while a grid whose progress lived
    in the messages would have to start again.

    The campaign and its results are read apart. Ordering, running and recording a cell need
    the campaign alone, which is small; reading a verdict, a selection or the figures needs
    every result, thousands of answers a cell, and asks for the reading.
    """

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        """The campaign stored under that identity, without what its cells produced.

        Raises:
            CampaignNotFoundError: If no campaign is stored under it.
        """
        ...

    def read(self, campaign_id: CampaignId) -> CampaignReading:
        """The campaign with the results of every cell it has recorded.

        Raises:
            CampaignNotFoundError: If no campaign is stored under it.
        """
        ...

    def get_result(self, campaign_id: CampaignId, cell: CampaignCell) -> CellResult:
        """What one recorded cell produced.

        Raises:
            CampaignNotFoundError: If no campaign is stored under that identity.
            UnknownCampaignCellError: If the campaign has not recorded that cell.
        """
        ...

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        """Store the campaign, provided nothing has changed it since revision ``seen``.

        A grid is worked by more than one process, and each of them reads the campaign, runs
        one cell and writes the campaign back. Writing over a state one never read would lose
        the cells another process recorded in between, so what a caller may write over is
        stated rather than assumed: the revision it read. A cell is recorded with its result
        through ``record``; this stores what else moves, the design once and the closing.

        Args:
            campaign: The campaign as the caller would now have it.
            seen: The revision the caller read, which is the state it is entitled to replace.
                A campaign nobody has stored yet is written whatever this says.

        Raises:
            CampaignChangedElsewhereError: If it has moved on since that revision.
            IncompleteCampaignError: If the campaign records a cell whose result is not stored.
        """
        ...

    def record(self, campaign: EvaluationCampaign, result: CellResult, *, seen: int) -> None:
        """Store the campaign with one more cell recorded, and what that cell produced, as one.

        Args:
            campaign: The campaign as the caller would now have it, the result's cell recorded.
            result: What the cell produced.
            seen: The revision the caller read.

        Raises:
            CampaignChangedElsewhereError: If the campaign has moved on since that revision.
            UnknownCampaignCellError: If the campaign does not record the result's cell.
            IncompleteCampaignError: If it records another cell whose result is not stored.
        """
        ...
