from sqlalchemy import Engine, Select, and_, func, or_, select, tuple_
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.campaign_cell_record import CampaignCellRecord
from emblema.evaluation.adapters.persistence.cell_result_rows import CellResultRows
from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import CampaignNotFoundError


class SqlAlchemyCampaignListing:
    """Pages of the campaign tables, by keyset over the order each table is read in.

    A page continues after a position and never from an offset, so a campaign designed or a
    cell recorded between two pages shifts nothing already read. A list of campaigns counts
    their cells in the same query instead of loading them: the cells, each with every unit's
    error, are the bulk of a campaign and a list shows none of them. The cells of one campaign
    are read a page at a time by their own key.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def campaigns(
        self, *, after: CampaignPosition | None, limit: int
    ) -> tuple[CampaignOverview, ...]:
        query = self._overviews().order_by(
            EvaluationCampaignRecord.opened_at.desc(), EvaluationCampaignRecord.id.asc()
        )
        if after is not None:
            opened_at = after.opened_at.value
            query = query.where(
                or_(
                    EvaluationCampaignRecord.opened_at < opened_at,
                    and_(
                        EvaluationCampaignRecord.opened_at == opened_at,
                        EvaluationCampaignRecord.id > after.campaign_id.value,
                    ),
                )
            )
        with Session(self._engine) as session:
            rows = session.execute(query.limit(limit)).all()
            return tuple(record.to_overview(cells) for record, cells in rows)

    def get_overview(self, campaign: CampaignId) -> CampaignOverview:
        query = self._overviews().where(EvaluationCampaignRecord.id == campaign.value)
        with Session(self._engine) as session:
            row = session.execute(query).one_or_none()
            if row is None:
                raise CampaignNotFoundError(f"no campaign stored under {campaign}")
            record, cells = row
            return record.to_overview(cells)

    @staticmethod
    def _overviews() -> Select[tuple[EvaluationCampaignRecord, int]]:
        """Campaigns without their cells, each beside how many cells it has recorded."""
        recorded = (
            select(func.count())
            .where(CampaignCellRecord.campaign_id == EvaluationCampaignRecord.id)
            .correlate(EvaluationCampaignRecord)
            .scalar_subquery()
        )
        return select(EvaluationCampaignRecord, recorded)

    def results(
        self, campaign: CampaignId, *, after: CampaignCell | None, limit: int
    ) -> tuple[CellResult, ...]:
        with Session(self._engine) as session:
            known = session.scalar(
                select(EvaluationCampaignRecord.id).where(
                    EvaluationCampaignRecord.id == campaign.value
                )
            )
            if known is None:
                raise CampaignNotFoundError(f"no campaign stored under {campaign}")
            query = (
                select(
                    CampaignCellRecord.candidate, CampaignCellRecord.budget, CampaignCellRecord.seed
                )
                .where(CampaignCellRecord.campaign_id == campaign.value)
                .order_by(
                    CampaignCellRecord.candidate, CampaignCellRecord.budget, CampaignCellRecord.seed
                )
            )
            if after is not None:
                query = query.where(
                    tuple_(
                        CampaignCellRecord.candidate,
                        CampaignCellRecord.budget,
                        CampaignCellRecord.seed,
                    )
                    > (str(after.candidate), after.budget.text(), after.seed)
                )
            keys = list(session.execute(query.limit(limit)).tuples())
            return CellResultRows(session).of_cells(campaign.value, keys)
