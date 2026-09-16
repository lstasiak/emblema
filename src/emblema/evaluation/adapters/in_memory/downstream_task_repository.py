from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import TaskNotFoundError
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class InMemoryDownstreamTaskRepository:
    """Keeps tasks in a dictionary, for a process that defines and draws in one go."""

    def __init__(self) -> None:
        self._tasks: dict[TaskId, DownstreamTask] = {}

    def get(self, task_id: TaskId) -> DownstreamTask:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise TaskNotFoundError(f"no task stored under {task_id}") from error

    def save(self, task: DownstreamTask) -> None:
        self._tasks[task.task_id] = task
