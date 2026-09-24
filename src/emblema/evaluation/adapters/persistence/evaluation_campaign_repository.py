from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    CampaignChangedElsewhereError,
    CampaignNotFoundError,
)


class SqlAlchemyEvaluationCampaignRepository:
    """Repository over the Evaluation schema of the metadata database.

    A campaign is read whole and written whole, its cells with it: what a worker needs in order
    to run a cell is the design, and what it needs in order to know whether the grid is now
    finished is every cell already recorded. Writing the whole state merges the cells by their
    own key, so a cell recorded twice is refused by the primary key rather than duplicated.

    Writing whole is why the revision is claimed first. Two workers that both read a grid, ran
    a cell each and wrote back would each write the cells it knew of, and the second would
    delete the first's. The row is locked and its revision compared before anything is written,
    so the check and the write cannot be interleaved and the loser is told rather than obeyed.
    The lock is held for the length of a save, which is a statement or two; a cell is minutes.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        with Session(self._engine) as session:
            record = session.get(EvaluationCampaignRecord, campaign_id.value)
            if record is None:
                raise CampaignNotFoundError(f"no campaign stored under {campaign_id}")
            return record.to_campaign()

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        with Session(self._engine) as session, session.begin():
            stored = session.execute(
                select(EvaluationCampaignRecord.version)
                .where(EvaluationCampaignRecord.id == campaign.campaign_id.value)
                .with_for_update()
            ).scalar_one_or_none()
            if stored is None:
                session.add(EvaluationCampaignRecord.from_campaign(campaign))
                return
            if stored != seen:
                raise CampaignChangedElsewhereError(
                    f"campaign {campaign.campaign_id} was read at revision {seen} and stands "
                    f"at {stored}"
                )
            session.merge(EvaluationCampaignRecord.from_campaign(campaign))
