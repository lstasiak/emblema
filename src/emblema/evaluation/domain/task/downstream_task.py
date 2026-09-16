from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import FrozenTestSplitClosedError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
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
