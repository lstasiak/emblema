from sqlalchemy import Engine
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import CampaignNotFoundError


class SqlAlchemyEvaluationCampaignRepository:
    """Repository over the Evaluation schema of the metadata database.

    A campaign is read whole and written whole, its cells with it: what a worker needs in order
    to run a cell is the design, and what it needs in order to know whether the grid is now
    finished is every cell already recorded. Writing the whole state merges the cells by their
    own key, so a cell recorded twice is refused by the primary key rather than duplicated.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        with Session(self._engine) as session:
            record = session.get(EvaluationCampaignRecord, campaign_id.value)
            if record is None:
                raise CampaignNotFoundError(f"no campaign stored under {campaign_id}")
            return record.to_campaign()

    def save(self, campaign: EvaluationCampaign) -> None:
        with Session(self._engine) as session, session.begin():
            session.merge(EvaluationCampaignRecord.from_campaign(campaign))
