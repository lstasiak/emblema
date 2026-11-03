from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    ForeignLabelSampleError,
    FrozenTestSplitClosedError,
    InvalidTaughtEventsError,
    MismatchedStratificationError,
    MixedTargetKindsError,
    ProtocolMismatchError,
    UnknownGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.label_scheme import LabelScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.stratification import Stratification
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.evaluation.domain.task.task_windows import TaskWindows
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class DownstreamTask:
    """A question asked of a corpus: which units answer it, under which protocol, and how.

    The task pins the corpus it was defined against by the reference to its manifest, so two
    tasks over two publications of the same data are two tasks. Its split is stored, not derived,
    and its label scheme and stratification are fixed here rather than at each run: both change
    which windows a budget draws, so either one chosen per run is a knob that can be turned once
    the numbers are visible.

    Invariants: a task that spends labels reads its target by a scheme of a quantity or an
    outcome and carries a stratification, since it cannot draw a budget without knowing how to
    read a label or how to spread it; a detection task reads the events its windows overlap and
    carries no stratification, since it draws no budget and teaches at most a few events; a
    binary target is spread over its outcomes and a quantity over ranks of the target, since
    bins of ranks cut through the middle of an outcome and outcomes do not exist on a scale.

    Attributes:
        task_id: Identity of the task.
        corpus: Name of the published corpus the units come from.
        manifest: Reference to the manifest the task was defined against.
        protocol: Which question the task asks, and so which quantity a campaign measures on it.
        split: Which units tune, which validate and which are held for the final run.
        labels: How a window's target is read from what the ground truth says about it; for a
            detection task, which event a scored window overlaps.
        strata: How a budget is spread over the pool; ``None`` where the protocol spends no
            labels.
        windows: Which of a unit's windows the task reads; every one unless the question is
            asked at one moment of the unit.
    """

    task_id: TaskId
    corpus: str
    manifest: ArtifactRef
    protocol: EvaluationProtocol
    split: TaskSplit
    labels: LabelScheme
    strata: Stratification | None
    windows: TaskWindows = TaskWindows.EVERY

    def __post_init__(self) -> None:
        detects = self.protocol is EvaluationProtocol.ANOMALY_DETECTION
        if (self.labels.kind is TargetKind.ANOMALY) != detects:
            raise ProtocolMismatchError(
                f"a task under {self.protocol} does not read its windows by {self.labels.kind}"
            )
        if (self.strata is None) == self.protocol.spends_labels:
            carries = "carries no" if self.strata is None else "carries a"
            raise ProtocolMismatchError(f"a task under {self.protocol} {carries} stratification")
        if self.strata is not None and (self.labels.kind is TargetKind.BINARY) != isinstance(
            self.strata, ClassStrata
        ):
            raise MismatchedStratificationError(
                f"a {self.labels.kind} target cannot be spread by {type(self.strata).__name__}"
            )

    @property
    def tuning_units(self) -> frozenset[UnitKey]:
        """Units a budget of labels may be drawn from."""
        return self.split.tuning

    @property
    def validation_units(self) -> frozenset[UnitKey]:
        """Units every number reported before the final run is measured on."""
        return self.split.validation

    def read_windows(self, windows: Iterable[TaskWindow]) -> tuple[TaskWindow, ...]:
        """The windows the task reads among those its units hold, in the order given.

        A budget is drawn from and a run is scored on these alone, so both read the unit at the
        same moment.
        """
        return self.windows.chosen_from(windows)

    def label_scheme(self) -> LabelScheme:
        """How this task's targets are read."""
        return self.labels

    def stratification(self) -> Stratification:
        """How a budget of this task's labels is spread over its pool.

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

        A detection task leaves out the windows its ground truth marks as not scored, such as
        those over a gap in the downlink: they carry no target and enter no reading.

        Raises:
            UnknownGroundTruthError: If ``truths`` says nothing about one of the windows.
            UnlabelledWindowError: If a window carries no label under the scheme.
        """
        scheme = self.label_scheme()
        labelled = []
        for window in windows:
            if window not in truths:
                raise UnknownGroundTruthError(
                    f"no ground truth known for unit {window.unit} at {window.position}"
                )
            target = self._target(scheme, window, truths[window])
            if target is not None:
                labelled.append(LabelledWindow(window=window, target=target))
        return tuple(labelled)

    @staticmethod
    def _target(scheme: LabelScheme, window: TaskWindow, truth: float) -> float | None:
        """What the scheme reads for the window; ``None`` for a window the task does not score."""
        match scheme:
            case RemainingLifeScheme():
                return scheme.target(failed_at=truth, ends_at=window.ends_at)
            case ForecastScheme():
                return scheme.target(exact=truth)
            case OutcomeScheme():
                return scheme.target(recorded=truth)
            case AnomalyScheme():
                return scheme.target(event=truth) if scheme.scores(truth) else None

    def accept_campaign(self, budgets: Iterable[LabelBudget]) -> None:
        """Refuse a campaign whose budgets of labels the task cannot spread its candidates over.

        A campaign spreads its candidates over budgets of labels. A task that spends none has
        no such axis: its one cell fits on every window of the fitting side, so the only budget
        it can run at is the one of every window, and a grid of several would read a curve off
        an axis that does not exist.

        Raises:
            ProtocolMismatchError: If the task spends no labels and the budgets are anything
                but the single budget of every window.
        """
        if not self.protocol.spends_labels and tuple(budgets) != (LabelBudget.everything(),):
            raise ProtocolMismatchError(
                f"a task under {self.protocol} has no budget of labels to spread a campaign "
                "over; its one cell fits on every window of the fitting side"
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

    def accept_taught(self, taught: Iterable[LabelledWindow]) -> None:
        """Refuse events a detector is taught that are not this task's to teach.

        A detection task spends no budget of labels, but a detector may be taught a counted
        number of the events its fitting side holds; the task, not the detector, says which
        windows those may be: the tuning side's, and nothing of the sides a run is scored on.

        Raises:
            ProtocolMismatchError: If the task spends labels, so it teaches no events.
            InvalidTaughtEventsError: If a taught window lies outside the tuning side.
        """
        if self.protocol.spends_labels:
            raise ProtocolMismatchError(
                f"a task under {self.protocol} draws a budget of labels and teaches no events"
            )
        for labelled in taught:
            if labelled.window.unit not in self.tuning_units:
                raise InvalidTaughtEventsError(
                    f"the taught window of unit {labelled.window.unit} is not on the tuning side "
                    f"of task {self.task_id}"
                )

    def accept_source(self, source: Self) -> None:
        """Refuse to learn this task from another task's labels of a different kind.

        A candidate fitted over several tasks pools their targets, each in its own scale; a
        probability and a quantity share no scale, so pooling them fits neither.

        Raises:
            MixedTargetKindsError: If the source's target is of another kind than this task's.
            ProtocolMismatchError: If either task spends no labels.
        """
        for named in (self, source):
            if not named.protocol.spends_labels:
                raise ProtocolMismatchError(
                    f"task {named.task_id} spends no labels and lends none to another task"
                )
        own, theirs = self.label_scheme().kind, source.label_scheme().kind
        if own is not theirs:
            raise MixedTargetKindsError(
                f"task {self.task_id} has a {own} target and cannot learn from the {theirs} "
                f"target of task {source.task_id}"
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
