from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignNotFoundError,
    UnknownCampaignCellError,
)


class InMemoryEvaluationCampaignRepository:
    """Keeps campaigns and their results in dictionaries, for a process that runs one in a go."""

    def __init__(self) -> None:
        self._campaigns: dict[CampaignId, EvaluationCampaign] = {}
        self._results: dict[CampaignId, dict[CampaignCell, CellResult]] = {}

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        try:
            return self._campaigns[campaign_id]
        except KeyError as error:
            raise CampaignNotFoundError(f"no campaign stored under {campaign_id}") from error

    def read(self, campaign_id: CampaignId) -> CampaignReading:
        campaign = self.get(campaign_id)
        results = self._results.get(campaign_id, {})
        return CampaignReading(
            campaign=campaign, results=tuple(results[cell] for cell in campaign.recorded)
        )

    def get_result(self, campaign_id: CampaignId, cell: CampaignCell) -> CellResult:
        self.get(campaign_id)
        try:
            return self._results.get(campaign_id, {})[cell]
        except KeyError as error:
            raise UnknownCampaignCellError(
                f"campaign {campaign_id} has not recorded {cell}"
            ) from error

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        self._claim(campaign, seen)
        campaign.accept_results(list(self._results.get(campaign.campaign_id, {})))
        self._campaigns[campaign.campaign_id] = campaign

    def record(self, campaign: EvaluationCampaign, result: CellResult, *, seen: int) -> None:
        self._claim(campaign, seen)
        if result.cell not in campaign.recorded:
            raise UnknownCampaignCellError(
                f"campaign {campaign.campaign_id} does not record {result.cell}"
            )
        stored = self._results.setdefault(campaign.campaign_id, {})
        campaign.accept_results([*stored, result.cell])
        stored[result.cell] = result
        self._campaigns[campaign.campaign_id] = campaign

    def stored(self) -> tuple[EvaluationCampaign, ...]:
        """Every campaign kept, in no particular order; what the in-memory listing reads."""
        return tuple(self._campaigns.values())

    def _claim(self, campaign: EvaluationCampaign, seen: int) -> None:
        stored = self._campaigns.get(campaign.campaign_id)
        if stored is not None and stored.revision != seen:
            raise CampaignChangedElsewhereError(
                f"campaign {campaign.campaign_id} was read at revision {seen} and stands at "
                f"{stored.revision}"
            )
