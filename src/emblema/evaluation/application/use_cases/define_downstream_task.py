from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import UnknownTaskUnitsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class DefineDownstreamTaskCommand:
    """Request to define a labelled task over part of a published corpus.

    Attributes:
        manifest: Manifest of the corpus the task draws its units from.
        units: Units the task covers; a corpus published from several subsets holds more than one
            task's worth, and which of them answer this question is a fact about the task.
        test: Units held for the final run, and where they come from.
        labels: How a window's target is read.
        strata: How many groups of the target a budget is spread over.
    """

    manifest: ArtifactRef
    units: frozenset[UnitKey]
    test: FrozenTestSplit
    labels: RemainingLifeScheme
    strata: TargetBins


class DefineDownstreamTask:
    """Defines a task whose sides are the corpus's own, and records them.

    The tuning and validation sides are the corpus's training and held-out units, narrowed to the
    ones the task covers, rather than a fresh draw: the corpus fitted its statistics on its
    training side and a backbone pretrained there has read those units unlabelled, so they are
    the wrong place to measure from. Both sides are stored with the task, so a later publication
    of the same corpus under another seed cannot move them.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        ids: IdGenerator,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._ids = ids

    def __call__(self, command: DefineDownstreamTaskCommand) -> TaskId:
        """Read the corpus's division, cut the task out of it and store the task.

        Raises:
            UnknownTaskUnitsError: If the task covers units the corpus does not name.
            InvalidTaskSplitError: If either side comes out empty or a unit sits on two sides.
            UnreadableTaskCorpusError: If the manifest is not one the port can read.
        """
        sides = self._corpus.describe(command.manifest)
        unknown = sorted(str(unit) for unit in command.units - (sides.training | sides.validation))
        if unknown:
            raise UnknownTaskUnitsError(f"corpus {sides.corpus!r} does not name units: {unknown}")
        task = DownstreamTask(
            task_id=self._ids.generate(TaskId),
            corpus=sides.corpus,
            manifest=command.manifest,
            split=TaskSplit(
                tuning=command.units & sides.training,
                validation=command.units & sides.validation,
                test=command.test,
            ),
            labels=command.labels,
            strata=command.strata,
        )
        self._tasks.save(task)
        return task.task_id
