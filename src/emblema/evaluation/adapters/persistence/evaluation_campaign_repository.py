from uuid import UUID

from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.campaign_cell_record import CampaignCellRecord
from emblema.evaluation.adapters.persistence.cell_result_rows import CellResultRows
from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
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


class SqlAlchemyEvaluationCampaignRepository:
    """Repository over the Evaluation schema of the metadata database.

    A campaign is one row and the keys of its recorded cells, which is all that ordering,
    running and recording a cell need. Its reading adds the rows of every cell's errors and
    answers, thousands a cell, read in three statements straight from the tables. A cell is
    written once, with its result, and never read back to be compared; the campaign's row moves
    only in its progress. A cell recorded twice is refused by the primary key as well as by the
    aggregate.

    The revision is claimed first. Two workers that both read a grid, ran a cell each and wrote
    back would each believe the other's cell absent. The row is locked and its revision compared
    before anything is written, so the check and the write cannot be interleaved and the loser
    is told rather than obeyed. The lock is held for the length of a save, which is the insert
    of one cell; a cell is minutes.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, campaign_id: CampaignId) -> EvaluationCampaign:
        with Session(self._engine) as session:
            return self._campaign(session, campaign_id)

    def read(self, campaign_id: CampaignId) -> CampaignReading:
        with Session(self._engine) as session:
            campaign = self._campaign(session, campaign_id)
            return CampaignReading(
                campaign=campaign, results=CellResultRows(session).of_campaign(campaign_id.value)
            )

    def get_result(self, campaign_id: CampaignId, cell: CampaignCell) -> CellResult:
        with Session(self._engine) as session:
            self._record(session, campaign_id)
            found = CellResultRows(session).of_cells(
                campaign_id.value, [CampaignCellRecord.key_of(cell)]
            )
            if not found:
                raise UnknownCampaignCellError(f"campaign {campaign_id} has not recorded {cell}")
            return found[0]

    def save(self, campaign: EvaluationCampaign, *, seen: int) -> None:
        with Session(self._engine) as session, session.begin():
            if self._claim(session, campaign, seen):
                return
            campaign.accept_results(self._recorded(session, campaign.campaign_id.value))
            self._progress(session, campaign)

    def record(self, campaign: EvaluationCampaign, result: CellResult, *, seen: int) -> None:
        if result.cell not in campaign.recorded:
            raise UnknownCampaignCellError(
                f"campaign {campaign.campaign_id} does not record {result.cell}"
            )
        with Session(self._engine) as session, session.begin():
            if self._claim(session, campaign, seen):
                return
            recorded = self._recorded(session, campaign.campaign_id.value)
            campaign.accept_results([*recorded, result.cell])
            session.add(CampaignCellRecord.from_result(campaign.campaign_id, result))
            self._progress(session, campaign)

    def _claim(self, session: Session, campaign: EvaluationCampaign, seen: int) -> bool:
        """Lock the campaign's row and compare its revision; store a new campaign whole.

        Returns:
            Whether the campaign was new and has now been stored, so nothing is left to write.

        Raises:
            CampaignChangedElsewhereError: If the row stands at another revision than ``seen``.
        """
        stored = session.execute(
            select(EvaluationCampaignRecord.version)
            .where(EvaluationCampaignRecord.id == campaign.campaign_id.value)
            .with_for_update()
        ).scalar_one_or_none()
        if stored is None:
            campaign.accept_results([])
            session.add(EvaluationCampaignRecord.from_campaign(campaign))
            return True
        if stored != seen:
            raise CampaignChangedElsewhereError(
                f"campaign {campaign.campaign_id} was read at revision {seen} and stands "
                f"at {stored}"
            )
        return False

    @staticmethod
    def _progress(session: Session, campaign: EvaluationCampaign) -> None:
        session.execute(
            update(EvaluationCampaignRecord)
            .where(EvaluationCampaignRecord.id == campaign.campaign_id.value)
            .values(**EvaluationCampaignRecord.progress_of(campaign))
        )

    def _campaign(self, session: Session, campaign_id: CampaignId) -> EvaluationCampaign:
        record = self._record(session, campaign_id)
        return record.to_campaign(self._recorded(session, campaign_id.value))

    @staticmethod
    def _record(session: Session, campaign_id: CampaignId) -> EvaluationCampaignRecord:
        record = session.get(EvaluationCampaignRecord, campaign_id.value)
        if record is None:
            raise CampaignNotFoundError(f"no campaign stored under {campaign_id}")
        return record

    @staticmethod
    def _recorded(session: Session, campaign_id: UUID) -> list[CampaignCell]:
        """The cells whose rows the campaign has, in the order of their keys."""
        rows = session.execute(
            select(CampaignCellRecord.candidate, CampaignCellRecord.budget, CampaignCellRecord.seed)
            .where(CampaignCellRecord.campaign_id == campaign_id)
            .order_by(
                CampaignCellRecord.candidate, CampaignCellRecord.budget, CampaignCellRecord.seed
            )
        ).tuples()
        return [
            CampaignCellRecord.cell_of(candidate, budget, seed) for candidate, budget, seed in rows
        ]
