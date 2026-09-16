from dataclasses import dataclass

from emblema.evaluation.contracts.events import FrozenTestSplitOpened
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.shared.events.domain_event import EventId
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class OpenTestSplitCommand:
    """Request to hand a task's frozen units to a run.

    Attributes:
        task: Task whose frozen side is wanted.
        purpose: What the run is for; only the final one may see it.
    """

    task: TaskId
    purpose: RunPurpose


class OpenTestSplit:
    """The only way to the frozen side, and the reason it leaves a trace.

    Nothing about publishing an event stops a determined caller, and that is not what it is for:
    the claim the project makes about its test set is that it was read once, and a claim of that
    shape is worth no more than the record behind it.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        ids: IdGenerator,
        clock: Clock,
        events: EventPublisher,
    ) -> None:
        self._tasks = tasks
        self._ids = ids
        self._clock = clock
        self._events = events

    def __call__(self, command: OpenTestSplitCommand) -> FrozenTestSplit:
        """Hand over the frozen side of a final run, and record that it happened.

        Raises:
            TaskNotFoundError: If the task is unknown.
            FrozenTestSplitClosedError: If the run is not the final one; nothing is recorded, as
                nothing was opened.
        """
        task = self._tasks.get(command.task)
        test = task.open_test_split(command.purpose)
        self._events.publish(
            FrozenTestSplitOpened(
                event_id=self._ids.generate(EventId),
                occurred_at=self._clock.now(),
                task=task.task_id,
                unit_count=len(test.units),
                source=test.source,
            )
        )
        return test
