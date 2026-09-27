from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, select, tuple_
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.campaign_cell_record import CampaignCellRecord
from emblema.evaluation.adapters.persistence.campaign_unit_error_record import (
    CampaignUnitErrorRecord,
)
from emblema.evaluation.adapters.persistence.campaign_window_prediction_record import (
    CampaignWindowPredictionRecord,
)
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction

type CellKey = tuple[str, str, int]


class CellResultRows:
    """Reads what cells produced straight from their rows, without an object per row.

    A cell's result is one row of the cell and thousands of rows of errors and answers. Loading
    those through the mapped classes costs an instance per row and a query per cell; here the
    three tables are each read in one statement, ordered by the cell's key, and the value
    objects are built from the columns as the mapped classes would build them. One session is
    read within, so the rows of one reading belong to one state.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def of_campaign(self, campaign_id: UUID) -> tuple[CellResult, ...]:
        """Every recorded cell's result, in the order of the cells' keys."""
        return self._results(campaign_id, None)

    def of_cells(self, campaign_id: UUID, keys: Sequence[CellKey]) -> tuple[CellResult, ...]:
        """The results of the cells at ``keys``, in key order; an unknown key yields none."""
        if not keys:
            return ()
        return self._results(campaign_id, keys)

    def _results(self, campaign_id: UUID, keys: Sequence[CellKey] | None) -> tuple[CellResult, ...]:
        errors: dict[CellKey, list[UnitError]] = {}
        for candidate, budget, seed, unit, squared_error, windows in self._session.execute(
            self._within(
                select(
                    CampaignUnitErrorRecord.candidate,
                    CampaignUnitErrorRecord.budget,
                    CampaignUnitErrorRecord.seed,
                    CampaignUnitErrorRecord.unit,
                    CampaignUnitErrorRecord.squared_error,
                    CampaignUnitErrorRecord.windows,
                ).order_by(
                    CampaignUnitErrorRecord.candidate,
                    CampaignUnitErrorRecord.budget,
                    CampaignUnitErrorRecord.seed,
                    CampaignUnitErrorRecord.unit,
                ),
                CampaignUnitErrorRecord,
                campaign_id,
                keys,
            )
        ).tuples():
            errors.setdefault((candidate, budget, seed), []).append(
                CampaignUnitErrorRecord.error_of(unit, squared_error, windows)
            )
        predictions: dict[CellKey, list[WindowPrediction]] = {}
        for (
            candidate,
            budget,
            seed,
            unit,
            position,
            ends_at,
            target,
            predicted,
        ) in self._session.execute(
            self._within(
                select(
                    CampaignWindowPredictionRecord.candidate,
                    CampaignWindowPredictionRecord.budget,
                    CampaignWindowPredictionRecord.seed,
                    CampaignWindowPredictionRecord.unit,
                    CampaignWindowPredictionRecord.position,
                    CampaignWindowPredictionRecord.ends_at,
                    CampaignWindowPredictionRecord.target,
                    CampaignWindowPredictionRecord.predicted,
                ).order_by(
                    CampaignWindowPredictionRecord.candidate,
                    CampaignWindowPredictionRecord.budget,
                    CampaignWindowPredictionRecord.seed,
                    CampaignWindowPredictionRecord.unit,
                    CampaignWindowPredictionRecord.position,
                ),
                CampaignWindowPredictionRecord,
                campaign_id,
                keys,
            )
        ).tuples():
            predictions.setdefault((candidate, budget, seed), []).append(
                CampaignWindowPredictionRecord.prediction_of(
                    unit, position, ends_at, target, predicted
                )
            )
        cells = self._session.execute(
            self._within(
                select(
                    CampaignCellRecord.candidate,
                    CampaignCellRecord.budget,
                    CampaignCellRecord.seed,
                    CampaignCellRecord.seconds,
                    CampaignCellRecord.artifact_key,
                    CampaignCellRecord.artifact_algorithm,
                    CampaignCellRecord.artifact_digest,
                ).order_by(
                    CampaignCellRecord.candidate, CampaignCellRecord.budget, CampaignCellRecord.seed
                ),
                CampaignCellRecord,
                campaign_id,
                keys,
            )
        ).tuples()
        results = []
        for candidate, budget, seed, seconds, key, algorithm, digest in cells:
            results.append(
                CampaignCellRecord.result_of(
                    candidate,
                    budget,
                    seed,
                    seconds,
                    key,
                    algorithm,
                    digest,
                    errors.get((candidate, budget, seed), ()),
                    predictions.get((candidate, budget, seed), ()),
                )
            )
        return tuple(results)

    @staticmethod
    def _within[T: tuple[object, ...]](
        query: Select[T],
        table: type[CampaignCellRecord]
        | type[CampaignUnitErrorRecord]
        | type[CampaignWindowPredictionRecord],
        campaign_id: UUID,
        keys: Sequence[CellKey] | None,
    ) -> Select[T]:
        query = query.where(table.campaign_id == campaign_id)
        if keys is not None:
            query = query.where(tuple_(table.candidate, table.budget, table.seed).in_(keys))
        return query
