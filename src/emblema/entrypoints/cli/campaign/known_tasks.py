"""The tasks this system can be told to define, each stated once.

A task is made of facts no adapter can read off a corpus: which of its units answer the
question, how a window's target is read, and which units are held for the one final run. A
register-class rather than a table in a file, because these are what a campaign and a report
refer to by name, and a task renamed in somebody's configuration would leave a finished grid
pointing at nothing.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Self

from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_scheme import LabelScheme
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.stratification import Stratification
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_windows import TaskWindows
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank


def _listed(listing: str) -> tuple[str, ...]:
    """The keys named one per line in ``listing``, a file shipped beside this module."""
    text = files(__package__).joinpath(listing).read_text(encoding="utf-8")
    return tuple(line for line in text.splitlines() if line)


@dataclass(frozen=True, kw_only=True)
class NamedTestUnits:
    """A frozen test side named outright: the units an official test set holds.

    Attributes:
        source: Where the units come from, as the set is known outside this system.
        units: Keys of the frozen test units.
    """

    source: str
    units: tuple[str, ...]

    @classmethod
    def listed(cls, source: str, listing: str) -> Self:
        """The units named one per line in ``listing``, a file shipped beside this module.

        For an official test set whose keys follow no range: the listing is the set as it was
        published, so the frozen side cannot drift from it however the corpus is read.
        """
        return cls(source=source, units=_listed(listing))

    def frozen_split(self, held_out: frozenset[UnitKey]) -> FrozenTestSplit:
        """The named units, whatever the corpus holds out: an official test set is not cut."""
        return FrozenTestSplit(
            units=frozenset(UnitKey(key) for key in self.units), source=self.source
        )


@dataclass(frozen=True, kw_only=True)
class HeldOutShare:
    """A frozen test side cut from the corpus's own held-out units, for a corpus without a test set.

    The held-out units are ranked under the seed and one in every ``one_in`` of them, rounded
    up, is frozen from the top of that ranking, so the same corpus always freezes the same units
    whatever order they are named in. A count rather than a fraction, so the arithmetic is exact.

    Attributes:
        source: Where the units come from, as the set is known outside this system.
        one_in: One unit in this many is frozen; at least two, so that some are left to validate on.
        seed: Seed of the ranking.
    """

    source: str
    one_in: int
    seed: int

    def __post_init__(self) -> None:
        if self.one_in < 2:
            raise ValueError(f"one_in must be at least two, got {self.one_in}")

    def frozen_split(self, held_out: frozenset[UnitKey]) -> FrozenTestSplit:
        ranked = sorted(held_out, key=lambda unit: seeded_rank(self.seed, str(unit)))
        count = -(-len(ranked) // self.one_in)
        return FrozenTestSplit(units=frozenset(ranked[:count]), source=self.source)


@dataclass(frozen=True, kw_only=True)
class KnownTask:
    """What a task is made of, as far as no adapter can read it off the corpus.

    Attributes:
        name: What the task is called in the reports.
        corpus: Name the corpus was published under.
        unit_prefix: What the keys of the task's units start with, inside that corpus.
        labels: How a window's target is read: the remaining life under a ceiling, the exact
            reading of a sensor a fixed time past the window, the outcome a unit recorded, or
            the event a window overlaps.
        strata: How a budget is spread over the pool: bins of the target's rank, or the two
            outcomes in proportion; ``None`` for a detection task, which draws no budget.
        units_called: What the task's units are called in prose, engines, units or stays.
        test: How the frozen test side is made; a task over part of a corpus keeps the frozen
            units its prefix names.
        windows: Which of a unit's windows the task reads.
        ineligible: Keys of units the prefix names that cannot answer the question, such as a
            stay too short for the window it is asked at or whose label has turned inside it.
        window_length: The length of the corpus's windows the task's ground truth assumes, in
            the corpus's unit of time; ``None`` where the truth is a fact about the unit.
    """

    name: str
    corpus: str
    unit_prefix: str
    labels: LabelScheme
    strata: Stratification | None
    units_called: str
    test: NamedTestUnits | HeldOutShare
    windows: TaskWindows = TaskWindows.EVERY
    ineligible: frozenset[str] = frozenset()
    window_length: float | None = None

    def units_of(self, keys: Sequence[str]) -> frozenset[UnitKey]:
        """The task's units among the keys a published corpus names."""
        return frozenset(
            UnitKey(key)
            for key in keys
            if key.startswith(self.unit_prefix) and key not in self.ineligible
        )

    def frozen_test(self, held_out: frozenset[UnitKey]) -> FrozenTestSplit:
        """The frozen test side, given the task's units the corpus holds out."""
        split = self.test.frozen_split(held_out)
        return FrozenTestSplit(
            units=frozenset(unit for unit in split.units if str(unit).startswith(self.unit_prefix)),
            source=split.source,
        )

    def defined_over(
        self, manifest: ArtifactRef, sides: CorpusSides
    ) -> DefineDownstreamTaskCommand:
        """This task as it is defined over a published corpus divided that way.

        The units a task covers cannot be named without the division: a corpus published from
        several subsets holds more than one task's worth, and which of them answer this question
        is what the prefix says. The frozen side is cut from the held-out units alone, and comes
        off the rest, so no unit is both learnt from and held for the final run.

        Raises:
            ValueError: If the corpus is windowed otherwise than the task's ground truth assumes,
                so its events would be placed against the wrong hours.
        """
        if self.window_length is not None and sides.window_length != self.window_length:
            raise ValueError(
                f"{self.name} reads events against windows of {self.window_length:g}, and the "
                f"corpus is published with windows of {sides.window_length:g}"
            )
        named = sorted(str(unit) for unit in sides.training | sides.validation)
        test = self.frozen_test(self.units_of(sorted(str(unit) for unit in sides.validation)))
        return DefineDownstreamTaskCommand(
            manifest=manifest,
            units=self.units_of(named) - test.units,
            test=test,
            protocol=self.protocol,
            labels=self.labels,
            strata=self.strata,
            windows=self.windows,
        )

    @property
    def protocol(self) -> EvaluationProtocol:
        """Which question the task asks, which its label scheme decides."""
        if isinstance(self.labels, AnomalyScheme):
            return EvaluationProtocol.ANOMALY_DETECTION
        return EvaluationProtocol.LABEL_BUDGET

    @property
    def labels_text(self) -> str:
        """The label scheme in a few words, for a heading."""
        match self.labels:
            case RemainingLifeScheme():
                return f"remaining life under a ceiling of {self.labels.ceiling:g}"
            case ForecastScheme():
                return (
                    f"the exact reading of {self.labels.channel} "
                    f"{self.labels.horizon:g} time units past the window"
                )
            case OutcomeScheme() if self.windows is TaskWindows.FIRST:
                return f"whether {self.labels.outcome} turned after each unit's first window"
            case OutcomeScheme():
                return f"the outcome {self.labels.outcome} each unit recorded"
            case AnomalyScheme():
                return "which annotated event, if any, each window overlaps"


class KnownTasks:
    """The tasks a report may run, each stated once."""

    TURBOFAN_FD001 = KnownTask(
        name="turbofan-fd001",
        corpus="cmapss",
        unit_prefix="FD001/",
        labels=RemainingLifeScheme(125.0),
        strata=TargetBins(4),
        units_called="engines",
        test=NamedTestUnits(
            source="cmapss/test/FD001",
            units=tuple(f"FD001/test/{engine}" for engine in range(1, 101)),
        ),
    )
    # The synthetic control's transfer leg: the forecasting task on the second layout of each
    # pair, a third of the held-out units frozen because a generated corpus has no test set.
    CONTROL_B_FORECAST = KnownTask(
        name="control-b-forecast",
        corpus="control-b",
        unit_prefix="control-b/",
        labels=ForecastScheme("s01", 12.0),
        strata=TargetBins(4),
        units_called="units",
        test=HeldOutShare(source="control-b/held-out", one_in=3, seed=1),
    )
    NULL_B_FORECAST = KnownTask(
        name="null-b-forecast",
        corpus="null-b",
        unit_prefix="null-b/",
        labels=ForecastScheme("s01", 12.0),
        strata=TargetBins(4),
        units_called="units",
        test=HeldOutShare(source="null-b/held-out", one_in=3, seed=1),
    )
    # The same task on the wide layouts, where the held-out side is large enough for the paired
    # interval to be narrower than the practical floor.
    CONTROL_B_WIDE_FORECAST = KnownTask(
        name="control-b-wide-forecast",
        corpus="control-b-wide",
        unit_prefix="control-b-wide/",
        labels=ForecastScheme("s01", 12.0),
        strata=TargetBins(4),
        units_called="units",
        test=HeldOutShare(source="control-b-wide/held-out", one_in=3, seed=1),
    )
    NULL_B_WIDE_FORECAST = KnownTask(
        name="null-b-wide-forecast",
        corpus="null-b-wide",
        unit_prefix="null-b-wide/",
        labels=ForecastScheme("s01", 12.0),
        strata=TargetBins(4),
        units_called="units",
        test=HeldOutShare(source="null-b-wide/held-out", one_in=3, seed=1),
    )
    # The ceiling: the second layout over the first's own trajectories, a leak by construction.
    CONTROL_B_SHARED_FORECAST = KnownTask(
        name="control-b-shared-forecast",
        corpus="control-b-shared",
        unit_prefix="control-b-shared/",
        labels=ForecastScheme("s01", 12.0),
        strata=TargetBins(4),
        units_called="units",
        test=HeldOutShare(source="control-b-shared/held-out", one_in=3, seed=1),
    )

    # Death in hospital after an intensive-care stay, the challenge's own question, read over
    # the first 48 hours of the stay; set C is the challenge's test set, named as published.
    PHYSIONET_IN_HOSPITAL_DEATH = KnownTask(
        name="physionet2012-in-hospital-death",
        corpus=KnownGroundTruths.PHYSIONET,
        unit_prefix="set-",
        labels=KnownGroundTruths.IN_HOSPITAL_DEATH,
        strata=ClassStrata(),
        units_called="stays",
        test=NamedTestUnits.listed("physionet2012/set-c", "physionet2012_set_c.txt"),
    )

    # Sepsis after the first recorded day of a stay in intensive care (PhysioNet/CinC Challenge
    # 2019), over both hospitals and over the second alone; the second is not the hospital the
    # stays of 2012 come from. The frozen side was cut from the corpus by a seeded rule
    # (scripts/sepsis_task_units.py at b872996e), since the challenge published no labelled test
    # set; the listings are the record, and the script was retired once they were published.
    PHYSIONET2019_SEPSIS = KnownTask(
        name="physionet2019-sepsis",
        corpus=KnownGroundTruths.PHYSIONET2019,
        unit_prefix="training_set",
        labels=KnownGroundTruths.SEPSIS,
        strata=ClassStrata(),
        units_called="patients",
        test=NamedTestUnits.listed("physionet2019/sepsis-frozen", "physionet2019_sepsis_test.txt"),
        windows=TaskWindows.FIRST,
        ineligible=frozenset(_listed("physionet2019_sepsis_ineligible.txt")),
    )
    PHYSIONET2019_SEPSIS_HOSPITAL_B = replace(
        PHYSIONET2019_SEPSIS, name="physionet2019-sepsis-hospital-b", unit_prefix="training_setB/"
    )

    # The satellite telemetry's events, one task per mission: the months of a mission's first
    # half are the corpus's training side and the detectors' fitting side, as the benchmark
    # trains; the months of its second half are held out, and one in three of them is frozen
    # for the final run, the rest validate. Anomalies and rare nominal events are events alike;
    # a window over a gap in the downlink is not scored.
    ESA_MISSION1_EVENTS = KnownTask(
        name="esa-mission1-events",
        corpus=KnownGroundTruths.ESA_AD_RAW,
        unit_prefix="ESA-Mission1/",
        labels=AnomalyScheme(),
        strata=None,
        units_called="months",
        test=HeldOutShare(source="esa_ad_raw/ESA-Mission1/second-half", one_in=3, seed=1),
        window_length=KnownGroundTruths.ESA_AD_RAW_WINDOW_HOURS,
    )
    ESA_MISSION2_EVENTS = replace(
        ESA_MISSION1_EVENTS,
        name="esa-mission2-events",
        unit_prefix="ESA-Mission2/",
        test=HeldOutShare(source="esa_ad_raw/ESA-Mission2/second-half", one_in=3, seed=1),
    )
    # The testbed's anomalies: the anomaly-free recording is the corpus's training side and the
    # detectors' fitting side, the experiments are held out, and one in three of them is frozen.
    SKAB_ANOMALIES = KnownTask(
        name="skab-anomalies",
        corpus=KnownGroundTruths.SKAB,
        unit_prefix="",
        labels=AnomalyScheme(),
        strata=None,
        units_called="experiments",
        test=HeldOutShare(source="skab/experiments", one_in=3, seed=1),
        window_length=KnownGroundTruths.SKAB_WINDOW_SECONDS,
    )
    # The telemetry fragments: the authors' training part is the corpus's training side and the
    # detectors' fitting side, their test part is held out, and one in three of it is frozen.
    OPSSAT_ANOMALIES = KnownTask(
        name="opssat-anomalies",
        corpus=KnownGroundTruths.OPSSAT_AD,
        unit_prefix="",
        labels=AnomalyScheme(),
        strata=None,
        units_called="fragments",
        test=HeldOutShare(source="opssat_ad/test", one_in=3, seed=1),
        window_length=KnownGroundTruths.OPSSAT_AD_WINDOW_SECONDS,
    )

    @classmethod
    def default(cls) -> KnownTask:
        return cls.TURBOFAN_FD001

    @classmethod
    def all(cls) -> tuple[KnownTask, ...]:
        return (
            cls.TURBOFAN_FD001,
            cls.CONTROL_B_FORECAST,
            cls.NULL_B_FORECAST,
            cls.CONTROL_B_WIDE_FORECAST,
            cls.NULL_B_WIDE_FORECAST,
            cls.CONTROL_B_SHARED_FORECAST,
            cls.PHYSIONET_IN_HOSPITAL_DEATH,
            cls.PHYSIONET2019_SEPSIS,
            cls.PHYSIONET2019_SEPSIS_HOSPITAL_B,
            cls.ESA_MISSION1_EVENTS,
            cls.ESA_MISSION2_EVENTS,
            cls.SKAB_ANOMALIES,
            cls.OPSSAT_ANOMALIES,
        )

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(task.name for task in cls.all())

    @classmethod
    def named(cls, name: str) -> KnownTask:
        """The task called ``name``.

        Raises:
            KeyError: If no task is called that.
        """
        for task in cls.all():
            if task.name == name:
                return task
        raise KeyError(name)
