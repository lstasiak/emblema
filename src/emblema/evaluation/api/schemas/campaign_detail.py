from typing import Self

from pydantic import BaseModel, Field

from emblema.evaluation.api.schemas.campaign_resource import CampaignResource
from emblema.evaluation.api.schemas.candidate_curve_resource import CandidateCurveResource
from emblema.evaluation.api.schemas.verdict_resource import VerdictResource
from emblema.evaluation.application.read_models.campaign_view import CampaignView


class CampaignDetail(BaseModel):
    """One campaign in full: the campaign, every candidate's curve, and the verdict if any."""

    campaign: CampaignResource
    curves: list[CandidateCurveResource]
    verdict: VerdictResource | None = Field(
        description="Absent while the grid runs, and for a selection, which concludes nothing."
    )

    @classmethod
    def of(cls, view: CampaignView) -> Self:
        return cls(
            campaign=CampaignResource.of(view.summary),
            curves=[CandidateCurveResource.of(curve) for curve in view.curves],
            verdict=None if view.verdict is None else VerdictResource.of(view.verdict),
        )
