from sqlalchemy import Engine, Select, and_, or_, select
from sqlalchemy.orm import Session

from emblema.serving.adapters.persistence.served_model_record import ServedModelRecord
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_position import ServedModelPosition
from emblema.serving.domain.served_model_state import ServedModelState


class SqlAlchemyServedModelListing:
    """Pages of the served models table, by keyset over promotion time and identity.

    The query says "rows after this position" and never "skip this many": an offset drifts when
    a model is promoted between two pages, a position does not, and it is read off the cursor
    rather than off a stored row, so a page costs one query.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def page(
        self, *, after: ServedModelPosition | None, limit: int, state: ServedModelState | None
    ) -> tuple[ServedModel, ...]:
        with Session(self._engine) as session:
            query = self._ordered(state)
            if after is not None:
                promoted_at = after.promoted_at.value
                query = query.where(
                    or_(
                        ServedModelRecord.promoted_at < promoted_at,
                        and_(
                            ServedModelRecord.promoted_at == promoted_at,
                            ServedModelRecord.id > after.served_model_id.value,
                        ),
                    )
                )
            records = session.scalars(query.limit(limit)).all()
            return tuple(record.to_model() for record in records)

    @staticmethod
    def _ordered(state: ServedModelState | None) -> Select[tuple[ServedModelRecord]]:
        query = select(ServedModelRecord).order_by(
            ServedModelRecord.promoted_at.desc(), ServedModelRecord.id.asc()
        )
        if state is ServedModelState.SERVING:
            query = query.where(ServedModelRecord.withdrawn_at.is_(None))
        elif state is ServedModelState.WITHDRAWN:
            query = query.where(ServedModelRecord.withdrawn_at.is_not(None))
        return query
