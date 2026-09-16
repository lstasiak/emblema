from typing import Protocol

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


class DownstreamTaskRepository(Protocol):
    """Keeps tasks, which is what makes a split a record rather than a derivation.

    A split that were recomputed whenever it was needed would move when the corpus was published
    again, when a seed changed, or when a unit was added — and every number measured before the
    move would quietly stop being comparable with the ones after it.
    """

    def get(self, task_id: TaskId) -> DownstreamTask:
        """The task stored under that identity.

        Raises:
            TaskNotFoundError: If no task is stored under it.
        """
        ...

    def save(self, task: DownstreamTask) -> None:
        """Store the task, replacing any earlier state of it."""
        ...
