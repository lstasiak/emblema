from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    ForeignLabelSampleError,
    FrozenTestSplitClosedError,
    UnknownUnitLifetimeError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class DownstreamTask:
    """A labelled question asked of a corpus: which units answer it, and how a label is read.

    The task pins the corpus it was defined against by the reference to its manifest, so two
    tasks over two publications of the same data are two tasks. Its split is stored, not derived,
    and its label scheme and stratification are fixed here rather than at each run: both change
    which windows a budget draws, so either one chosen per run is a knob that can be turned once
    the numbers are visible.

    Attributes:
        task_id: Identity of the task.
        corpus: Name of the published corpus the units come from.
        manifest: Reference to the manifest the task was defined against.
        split: Which units tune, which validate and which are held for the final run.
        labels: How a window's target is read from the moment its unit failed.
        strata: How many groups of the target a budget is spread over.
    """

    task_id: TaskId
    corpus: str
    manifest: ArtifactRef
    split: TaskSplit
    labels: RemainingLifeScheme
    strata: TargetBins

    @property
    def tuning_units(self) -> frozenset[UnitKey]:
        """Units a budget of labels may be drawn from."""
        return self.split.tuning

    @property
    def validation_units(self) -> frozenset[UnitKey]:
        """Units every number reported before the final run is measured on."""
        return self.split.validation

    def labelled(
        self, windows: Iterable[TaskWindow], failed_at: Mapping[UnitKey, float]
    ) -> tuple[LabelledWindow, ...]:
        """Those windows with the target the task's scheme reads for each, in the order given.

        Raises:
            UnknownUnitLifetimeError: If a window's unit has no failure time in ``failed_at``.
            UnlabelledWindowError: If a window reaches past the failure of its unit.
        """
        return tuple(self._labelled(window, failed_at) for window in windows)

    def _labelled(self, window: TaskWindow, failed_at: Mapping[UnitKey, float]) -> LabelledWindow:
        if window.unit not in failed_at:
            raise UnknownUnitLifetimeError(f"no failure time known for unit {window.unit}")
        return LabelledWindow(
            window=window,
            target=self.labels.target(failed_at=failed_at[window.unit], ends_at=window.ends_at),
        )

    def accept_sample(self, sample: LabelSample) -> None:
        """Refuse a sample of labels drawn from another task.

        A sample carries the identity of the task it was drawn from, and a candidate learnt from
        one task's labels and scored on another's windows would measure nothing anyone asked.

        Raises:
            ForeignLabelSampleError: If the sample names another task.
        """
        if sample.task != self.task_id:
            raise ForeignLabelSampleError(
                f"the sample was drawn from task {sample.task}, not {self.task_id}"
            )

    def open_test_split(self, purpose: RunPurpose) -> FrozenTestSplit:
        """The frozen side, for a run that is allowed to see it.

        Raises:
            FrozenTestSplitClosedError: If the run is not the final one.
        """
        if purpose is not RunPurpose.FINAL:
            raise FrozenTestSplitClosedError(
                f"the test side of task {self.task_id} is frozen for a {purpose.value} run"
            )
        return self.split.test
