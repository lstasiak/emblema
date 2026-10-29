from dataclasses import dataclass

from emblema.evaluation.application.use_cases.open_test_split import (
    OpenTestSplit,
    OpenTestSplitCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.detection.detection_windows import DetectionWindows
from emblema.evaluation.domain.exceptions import ProtocolMismatchError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.ground_truth import GroundTruth


@dataclass(frozen=True, kw_only=True)
class DrawDetectionWindowsCommand:
    """Which detection task a run is over, and what the run is for.

    Attributes:
        task: Task to fit on and to be scored on.
        purpose: What the run is for, which decides which side it is scored on.
    """

    task: TaskId
    purpose: RunPurpose = RunPurpose.TUNING


class DrawDetectionWindows:
    """Reads the windows a detector fits on, without a label, and labels the side it is scored on.

    The counterpart of drawing a run's labels, for a task that spends none. The fitting side is
    the task's tuning side whole, read as windows alone: no label reaches the fit, which is what
    makes the detector a zero-label one. A tuning run is scored on the validation units, labelled
    by the events they overlap, and never asks for the frozen ones; the final run asks for them
    through the operation that records the asking. A selection run is refused: a detector's
    settings are declared before the run, since choosing them on labels of the tuning side would
    spend labels the protocol says it has not got.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        truth: GroundTruth,
        open_test_split: OpenTestSplit,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._truth = truth
        self._open = open_test_split

    def __call__(self, command: DrawDetectionWindowsCommand) -> DetectionWindows:
        """The task, the windows of its fitting side, and the labelled side it answers.

        Raises:
            TaskNotFoundError: If the task is unknown.
            ProtocolMismatchError: If the task spends labels, or the run is a selection.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a scored window.
            InvalidScoredOutcomeError: If either side holds no window.
        """
        task = self._tasks.get(command.task)
        if task.protocol.spends_labels:
            raise ProtocolMismatchError(
                f"task {task.task_id} spends labels; a detector fits on none"
            )
        scored_units = self._scored_units(task, command.purpose)
        fitting = task.read_windows(self._corpus.windows_of(task.manifest, task.tuning_units))
        windows = task.read_windows(self._corpus.windows_of(task.manifest, scored_units))
        scored = task.labelled(windows, self._truth.truths_of(task.corpus, windows))
        return DetectionWindows(task=task, fitting=fitting, scored=scored)

    def _scored_units(self, task: DownstreamTask, purpose: RunPurpose) -> frozenset[UnitKey]:
        match purpose:
            case RunPurpose.SELECTION:
                raise ProtocolMismatchError(
                    "a detector's settings are declared before the run; nothing is selected on "
                    "the tuning side of a task that spends no labels"
                )
            case RunPurpose.TUNING:
                return task.validation_units
            case _:
                return self._open(OpenTestSplitCommand(task=task.task_id, purpose=purpose)).units
