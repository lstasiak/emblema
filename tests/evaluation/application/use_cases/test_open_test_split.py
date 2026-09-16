from datetime import UTC, datetime

import pytest

from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.application.use_cases.open_test_split import (
    OpenTestSplit,
    OpenTestSplitCommand,
)
from emblema.evaluation.contracts.events import FrozenTestSplitOpened
from emblema.evaluation.domain.exceptions import FrozenTestSplitClosedError, TaskNotFoundError
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import TASK, TEST_SIDE, task

AT = UtcDateTime(datetime(2026, 9, 16, 9, 0, tzinfo=UTC))


def opening(stored: bool = True) -> tuple[OpenTestSplit, list[FrozenTestSplitOpened]]:
    tasks = InMemoryDownstreamTaskRepository()
    if stored:
        tasks.save(task())
    subscriber = InMemoryEventSubscriber()
    seen: list[FrozenTestSplitOpened] = []
    subscriber.subscribe(FrozenTestSplitOpened, seen.append)
    publisher = InMemoryEventPublisher(subscriber)
    return OpenTestSplit(tasks, SequentialIdGenerator(), FixedClock(AT), publisher), seen


def test_the_final_run_is_handed_the_frozen_units() -> None:
    use_case, _ = opening()

    assert use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.FINAL)) == TEST_SIDE


def test_opening_the_frozen_side_is_recorded() -> None:
    use_case, seen = opening()

    use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.FINAL))

    assert [(event.task, event.unit_count, event.source, event.occurred_at) for event in seen] == [
        (TASK, 1, "turbofans/test", AT)
    ]


def test_every_opening_is_recorded_not_only_the_first() -> None:
    use_case, seen = opening()

    use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.FINAL))
    use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.FINAL))

    assert len(seen) == 2


def test_a_tuning_run_is_refused_and_leaves_no_record() -> None:
    use_case, seen = opening()

    with pytest.raises(FrozenTestSplitClosedError):
        use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.TUNING))

    assert seen == []


def test_opening_a_task_nobody_defined_is_refused() -> None:
    use_case, _ = opening(stored=False)

    with pytest.raises(TaskNotFoundError):
        use_case(OpenTestSplitCommand(task=TASK, purpose=RunPurpose.FINAL))
