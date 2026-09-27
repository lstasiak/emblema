from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.campaign.cell_result import CellResult


class InMemoryCampaignListing:
    """Lists what the in-memory repository holds, sorted and cut the way the database sorts."""

    def __init__(self, repository: InMemoryEvaluationCampaignRepository) -> None:
        self._repository = repository

    def campaigns(
        self, *, after: CampaignPosition | None, limit: int
    ) -> tuple[CampaignOverview, ...]:
        ordered = sorted(
            (CampaignOverview.of(campaign) for campaign in self._repository.stored()),
            key=lambda overview: self._key(overview.position),
        )
        if after is not None:
            start = self._key(after)
            ordered = [overview for overview in ordered if self._key(overview.position) > start]
        return tuple(ordered[:limit])

    def get_overview(self, campaign: CampaignId) -> CampaignOverview:
        return CampaignOverview.of(self._repository.get(campaign))

    def results(
        self, campaign: CampaignId, *, after: CampaignCell | None, limit: int
    ) -> tuple[CellResult, ...]:
        ordered = sorted(self._repository.read(campaign).results, key=self._cell_position)
        if after is not None:
            start = self._key_of(after)
            ordered = [result for result in ordered if self._cell_position(result) > start]
        return tuple(ordered[:limit])

    @staticmethod
    def _key(position: CampaignPosition) -> tuple[float, str]:
        # Latest design first: the timestamp is negated so one ascending key serves both.
        return (-position.opened_at.value.timestamp(), str(position.campaign_id))

    @classmethod
    def _cell_position(cls, result: CellResult) -> tuple[str, str, int]:
        return cls._key_of(result.cell)

    @staticmethod
    def _key_of(cell: CampaignCell) -> tuple[str, str, int]:
        return (str(cell.candidate), cell.budget.text(), cell.seed)
