from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    ForeignLabelSampleError,
    FrozenTestSplitClosedError,
    ProtocolMismatchError,
    UnknownGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class DownstreamTask:
    """A question asked of a corpus: which units answer it, under which protocol, and how.

    The task pins the corpus it was defined against by the reference to its manifest, so two
    tasks over two publications of the same data are two tasks. Its split is stored, not derived,
    and its label scheme and stratification are fixed here rather than at each run: both change
    which windows a budget draws, so either one chosen per run is a knob that can be turned once
    the numbers are visible.

    Invariants: the label scheme and the stratification are present exactly when the protocol
    spends labels — a detection task has no target to read per window, and a supervised one
    cannot draw a budget without knowing how to read one.

    Attributes:
        task_id: Identity of the task.
        corpus: Name of the published corpus the units come from.
        manifest: Reference to the manifest the task was defined against.
        protocol: Which question the task asks, and so which quantity a campaign measures on it.
        split: Which units tune, which validate and which are held for the final run.
        labels: How a window's target is read from what the ground truth says about it: the
            moment its unit failed, or the exact reading it is asked to forecast. A closed set
            of schemes, because which one a task uses decides what the ground truth has to say;
            ``None`` where the protocol spends no labels.
        strata: How many groups of the target a budget is spread over; ``None`` where the
            protocol spends no labels.
    """

    task_id: TaskId
    corpus: str
    manifest: ArtifactRef
    protocol: EvaluationProtocol
    split: TaskSplit
    labels: RemainingLifeScheme | ForecastScheme | None
    strata: TargetBins | None

    def __post_init__(self) -> None:
        for named, part in (("label scheme", self.labels), ("stratification", self.strata)):
            if (part is None) == self.protocol.spends_labels:
                carries = "carries no" if part is None else "carries a"
                raise ProtocolMismatchError(f"a task under {self.protocol} {carries} {named}")

    @property
    def tuning_units(self) -> frozenset[UnitKey]:
        """Units a budget of labels may be drawn from."""
        return self.split.tuning

    @property
    def validation_units(self) -> frozenset[UnitKey]:
        """Units every number reported before the final run is measured on."""
        return self.split.validation

    def label_scheme(self) -> RemainingLifeScheme | ForecastScheme:
        """How this task's targets are read.

        Raises:
            ProtocolMismatchError: If the protocol spends no labels, so there is no scheme.
        """
        if self.labels is None:
            raise ProtocolMismatchError(f"a task under {self.protocol} reads no label per window")
        return self.labels

    def stratification(self) -> TargetBins:
        """How many groups of the target a budget of this task's labels is spread over.

        Raises:
            ProtocolMismatchError: If the protocol spends no labels, so no budget is drawn.
        """
        if self.strata is None:
            raise ProtocolMismatchError(f"a task under {self.protocol} draws no budget of labels")
        return self.strata

    def labelled(
        self, windows: Iterable[TaskWindow], truths: Mapping[TaskWindow, float]
    ) -> tuple[LabelledWindow, ...]:
        """Those windows with the target the task's scheme reads for each, in the order given.

        Raises:
            ProtocolMismatchError: If the protocol spends no labels.
            UnknownGroundTruthError: If ``truths`` says nothing about one of the windows.
            UnlabelledWindowError: If a window reaches past the failure of its unit.
        """
        scheme = self.label_scheme()
        return tuple(self._labelled(scheme, window, truths) for window in windows)

    @staticmethod
    def _labelled(
        scheme: RemainingLifeScheme | ForecastScheme,
        window: TaskWindow,
        truths: Mapping[TaskWindow, float],
    ) -> LabelledWindow:
        if window not in truths:
            raise UnknownGroundTruthError(
                f"no ground truth known for unit {window.unit} at {window.position}"
            )
        match scheme:
            case RemainingLifeScheme():
                target = scheme.target(failed_at=truths[window], ends_at=window.ends_at)
            case ForecastScheme():
                target = scheme.target(exact=truths[window])
        return LabelledWindow(window=window, target=target)

    def accept_campaign(self) -> None:
        """Refuse a campaign over a task that spends no labels.

        A campaign spreads its candidates over budgets of labels, so a task that reads no label
        per window offers it no axis to spread them over and no curve to read off the result.

        Raises:
            ProtocolMismatchError: If the protocol spends no labels.
        """
        if not self.protocol.spends_labels:
            raise ProtocolMismatchError(
                f"a task under {self.protocol} has no budget of labels to spread a campaign over"
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
