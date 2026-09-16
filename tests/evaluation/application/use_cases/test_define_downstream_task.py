import pytest

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTask,
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidTaskSplitError, UnknownTaskUnitsError
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.evaluation.support import MANIFEST, SCHEME, STRATA, TEST_SIDE, sides, units

PUBLISHED = sides(training=units("a", "b", "x"), validation=units("c", "y"))


def define(
    covering: frozenset = units("a", "b", "c"),
) -> tuple[InMemoryDownstreamTaskRepository, TaskId]:
    tasks = InMemoryDownstreamTaskRepository()
    use_case = DefineDownstreamTask(
        tasks, InMemoryCorpusWindows(PUBLISHED, {}), SequentialIdGenerator()
    )
    task_id = use_case(
        DefineDownstreamTaskCommand(
            manifest=MANIFEST, units=covering, test=TEST_SIDE, labels=SCHEME, strata=STRATA
        )
    )
    return tasks, task_id


def test_the_task_inherits_the_sides_the_corpus_already_drew() -> None:
    tasks, task_id = define()

    task = tasks.get(task_id)
    assert task.tuning_units == units("a", "b")
    assert task.validation_units == units("c")


def test_the_task_covers_only_the_units_it_was_defined_over() -> None:
    tasks, task_id = define(covering=units("a", "c"))

    assert tasks.get(task_id).tuning_units == units("a")


def test_the_task_records_the_corpus_and_the_manifest_it_was_cut_from() -> None:
    tasks, task_id = define()

    task = tasks.get(task_id)
    assert (task.corpus, task.manifest) == ("turbofans", MANIFEST)


def test_units_the_corpus_does_not_name_are_refused() -> None:
    with pytest.raises(UnknownTaskUnitsError, match="does not name units"):
        define(covering=units("a", "c", "ghost"))


def test_a_task_with_nothing_to_validate_on_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="validation side"):
        define(covering=units("a", "b"))


def test_a_task_whose_frozen_units_also_tune_it_is_refused() -> None:
    tasks = InMemoryDownstreamTaskRepository()
    use_case = DefineDownstreamTask(
        tasks, InMemoryCorpusWindows(PUBLISHED, {}), SequentialIdGenerator()
    )
    held = FrozenTestSplit(units=units("a"), source="turbofans/test")

    with pytest.raises(InvalidTaskSplitError, match="tuning and test"):
        use_case(
            DefineDownstreamTaskCommand(
                manifest=MANIFEST,
                units=units("a", "b", "c"),
                test=held,
                labels=SCHEME,
                strata=STRATA,
            )
        )
