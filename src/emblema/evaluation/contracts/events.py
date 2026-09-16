from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.shared.events.domain_event import DomainEvent


@dataclass(frozen=True, kw_only=True)
class FrozenTestSplitOpened(DomainEvent):
    """The frozen side of a task was handed to a run.

    This is published every time it happens, not only the first, so that the record shows whether
    the promise of a single final run was kept. A project that cannot say how often it read its
    own test set is not in a position to claim it read it once.

    Attributes:
        task: Task whose frozen side was opened.
        unit_count: How many units were handed over.
        source: Where those units come from, as the set is known outside this system.
    """

    task: TaskId
    unit_count: int
    source: str
