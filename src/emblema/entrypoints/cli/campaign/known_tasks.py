"""The tasks this system can be told to define, each stated once.

A task is made of facts no adapter can read off a corpus: which of its units answer the
question, how a window's target is read, and which units are held for the one final run. A
register-class rather than a table in a file, because these are what a campaign and a report
refer to by name, and a task renamed in somebody's configuration would leave a finished grid
pointing at nothing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank


@dataclass(frozen=True, kw_only=True)
class NamedTestUnits:
    """A frozen test side named outright: the units an official test set holds.

    Attributes:
        source: Where the units come from, as the set is known outside this system.
        units: Keys of the frozen test units.
    """

    source: str
    units: tuple[str, ...]

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
    """What a supervised task is made of, as far as no adapter can read it off the corpus.

    Attributes:
        name: What the task is called in the reports.
        corpus: Name the corpus was published under.
        unit_prefix: What the keys of the task's units start with, inside that corpus.
        labels: How a window's target is read: the remaining life under a ceiling, or the exact
            reading of a sensor a fixed time past the window.
        strata: How many groups of the target a budget is spread over.
        units_called: What the task's units are called in prose, engines or units.
        test: How the frozen test side is made.
    """

    name: str
    corpus: str
    unit_prefix: str
    labels: RemainingLifeScheme | ForecastScheme
    strata: int
    units_called: str
    test: NamedTestUnits | HeldOutShare

    def units_of(self, keys: Sequence[str]) -> frozenset[UnitKey]:
        """The task's units among the keys a published corpus names."""
        return frozenset(UnitKey(key) for key in keys if key.startswith(self.unit_prefix))

    def frozen_test(self, held_out: frozenset[UnitKey]) -> FrozenTestSplit:
        """The frozen test side, given the task's units the corpus holds out."""
        return self.test.frozen_split(held_out)

    def defined_over(
        self, manifest: ArtifactRef, sides: CorpusSides
    ) -> DefineDownstreamTaskCommand:
        """This task as it is defined over a published corpus divided that way.

        The units a task covers cannot be named without the division: a corpus published from
        several subsets holds more than one task's worth, and which of them answer this question
        is what the prefix says. The frozen side is cut from the held-out units alone, and comes
        off the rest, so no unit is both learnt from and held for the final run.
        """
        named = sorted(str(unit) for unit in sides.training | sides.validation)
        test = self.frozen_test(self.units_of(sorted(str(unit) for unit in sides.validation)))
        return DefineDownstreamTaskCommand(
            manifest=manifest,
            units=self.units_of(named) - test.units,
            test=test,
            protocol=EvaluationProtocol.LABEL_BUDGET,
            labels=self.labels,
            strata=TargetBins(self.strata),
        )

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


class KnownTasks:
    """The tasks a report may run, each stated once."""

    TURBOFAN_FD001 = KnownTask(
        name="turbofan-fd001",
        corpus="cmapss",
        unit_prefix="FD001/",
        labels=RemainingLifeScheme(125.0),
        strata=4,
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
        strata=4,
        units_called="units",
        test=HeldOutShare(source="control-b/held-out", one_in=3, seed=1),
    )
    NULL_B_FORECAST = KnownTask(
        name="null-b-forecast",
        corpus="null-b",
        unit_prefix="null-b/",
        labels=ForecastScheme("s01", 12.0),
        strata=4,
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
        strata=4,
        units_called="units",
        test=HeldOutShare(source="control-b-wide/held-out", one_in=3, seed=1),
    )
    NULL_B_WIDE_FORECAST = KnownTask(
        name="null-b-wide-forecast",
        corpus="null-b-wide",
        unit_prefix="null-b-wide/",
        labels=ForecastScheme("s01", 12.0),
        strata=4,
        units_called="units",
        test=HeldOutShare(source="null-b-wide/held-out", one_in=3, seed=1),
    )
    # The ceiling: the second layout over the first's own trajectories, a leak by construction.
    CONTROL_B_SHARED_FORECAST = KnownTask(
        name="control-b-shared-forecast",
        corpus="control-b-shared",
        unit_prefix="control-b-shared/",
        labels=ForecastScheme("s01", 12.0),
        strata=4,
        units_called="units",
        test=HeldOutShare(source="control-b-shared/held-out", one_in=3, seed=1),
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
