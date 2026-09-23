from sqlalchemy import Engine
from sqlalchemy.orm import Session

from emblema.evaluation.adapters.persistence.downstream_task_record import DownstreamTaskRecord
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import TaskNotFoundError
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class SqlAlchemyDownstreamTaskRepository:
    """Repository over the Evaluation schema of the metadata database.

    A task is read whole and written whole: ``save`` merges the record of the state the caller
    holds, so the row and its units are inserted or updated by identity in one transaction.
    Nothing here is unique but the identity, so no integrity failure has a domain meaning and
    any one propagates.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, task_id: TaskId) -> DownstreamTask:
        with Session(self._engine) as session:
            record = session.get(DownstreamTaskRecord, task_id.value)
            if record is None:
                raise TaskNotFoundError(f"no task stored under {task_id}")
            return record.to_task()

    def save(self, task: DownstreamTask) -> None:
        with Session(self._engine) as session, session.begin():
            session.merge(DownstreamTaskRecord.from_task(task))
