"""One task, stated once, for every test that needs a task rather than a rule."""

from uuid import UUID

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

CORPUS = "turbofans"
MANIFEST = ArtifactRef(key="durable/manifest", checksum=Checksum.of_bytes(b"manifest"))
TASK = TaskId(UUID(int=1))
SCHEME = RemainingLifeScheme(125.0)
STRATA = TargetBins(4)
TEST_SIDE = FrozenTestSplit(units=frozenset({UnitKey("held/1")}), source="turbofans/test")


def units(*names: str) -> frozenset[UnitKey]:
    return frozenset(UnitKey(name) for name in names)


def sides(training: frozenset[UnitKey], validation: frozenset[UnitKey]) -> CorpusSides:
    return CorpusSides(corpus=CORPUS, training=training, validation=validation)


def window(unit: str, position: int, ends_at: float) -> TaskWindow:
    return TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at)


def task(
    tuning: frozenset[UnitKey] = units("a", "b"),
    validation: frozenset[UnitKey] = units("c"),
    test: FrozenTestSplit = TEST_SIDE,
) -> DownstreamTask:
    return DownstreamTask(
        task_id=TASK,
        corpus=CORPUS,
        manifest=MANIFEST,
        split=TaskSplit(tuning=tuning, validation=validation, test=test),
        labels=SCHEME,
        strata=STRATA,
    )
