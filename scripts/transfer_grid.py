"""The grid of the label-efficiency curve as it is stored: its cells, its files, its task.

A cell is a transfer mode at a budget of labels under a seed. The transfer report writes cells
into a directory as three CSV files — a row per run, per epoch and per validation window — and
the curve report reads such directories back; what both need to agree on is here, free of the
training stack, so that reading a grid never imports torch.
"""

import csv
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Self

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.ports.artifact_store import ArtifactStore

RUNS, EPOCHS, PREDICTIONS = "runs.csv", "epochs.csv", "predictions.csv"
# What a plan flattens to, in the order ``AdaptationPlan.parameters`` states it.
PLAN_COLUMNS = (
    "mode",
    "backbone",
    "run_seed",
    "epochs",
    "min_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_fraction",
    "final_lr_fraction",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
)
RUN_COLUMNS = (
    *PLAN_COLUMNS,
    "task",
    "budget",
    "windows",
    "steps",
    "sample_seed",
    "engines",
    "trainable_parameters",
    "rmse",
    "seconds",
    "device",
    "commit",
    "torch",
)
# The cell a row belongs to, the columns that tell one cell from another.
CELL_COLUMNS = ("mode", "budget", "seed")
BUDGETS = ("50", "200", "1000", "all")
SEEDS = (1, 2, 3, 4, 5)
MODES = tuple(str(mode) for mode in TransferMode)


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

    @classmethod
    def default(cls) -> KnownTask:
        return cls.TURBOFAN_FD001

    @classmethod
    def all(cls) -> tuple[KnownTask, ...]:
        return (cls.TURBOFAN_FD001, cls.CONTROL_B_FORECAST, cls.NULL_B_FORECAST)

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


def budget_of(text: str) -> LabelBudget:
    return LabelBudget.everything() if text == "all" else LabelBudget.of(int(text))


def budget_text(budget: LabelBudget) -> str:
    return "all" if budget.windows is None else str(budget.windows)


def budget_rank(budget: str) -> float:
    """Where a budget sorts: by its count of windows, everything last."""
    return float("inf") if budget == "all" else float(int(budget))


@dataclass(frozen=True, kw_only=True)
class Cell:
    """One point of the grid: a mode at a budget under a seed."""

    mode: TransferMode
    budget: LabelBudget
    seed: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (str(self.mode), budget_text(self.budget), str(self.seed))

    def __str__(self) -> str:
        return f"{self.mode} at {budget_text(self.budget)} under seed {self.seed}"


@dataclass(frozen=True, kw_only=True)
class Stored:
    """Where one report's cells are kept, and how a cell is appended to them.

    One directory holds one grid or a part of it. A cell's epochs and predictions are keyed by
    the cell, and its row in the runs file is written last, as the mark that the cell is whole:
    a directory reopened after a dropped session holds exactly the cells that have such a row,
    and discards the epochs and predictions of a cell that has none.
    """

    directory: Path

    @classmethod
    def open(cls, directory: Path) -> Self:
        """The directory a report runs into, made if missing, its uncommitted cells dropped."""
        directory.mkdir(parents=True, exist_ok=True)
        stored = cls(directory=directory)
        stored._discard_uncommitted()
        return stored

    @classmethod
    def existing(cls, directory: Path) -> Self:
        """A report already stored, to be read again.

        Raises:
            SystemExit: If nothing is stored there.
        """
        if not (directory / RUNS).is_file():
            raise SystemExit(f"{directory} holds no {RUNS}; nothing to render")
        return cls(directory=directory)

    def holds(self, cell: Cell, plan: AdaptationPlan, *, task: str, commit: str) -> bool:
        """Whether the cell is stored under this plan, this task and this commit.

        Raises:
            SystemExit: If the cell is stored under another plan or commit: a grid resumed
                across configurations would mix cells that are not comparable.
        """
        stated = {name: str(value) for name, value in plan.parameters().items()}
        for row in self.runs():
            if (row["mode"], row["budget"], row["sample_seed"]) != cell.key:
                continue
            differing = [name for name, value in stated.items() if row.get(name) != value]
            if row.get("task") != task:
                differing.append("task")
            if row["commit"] != commit:
                differing.append("commit")
            if differing:
                raise SystemExit(
                    f"{self.directory} holds {cell} under another configuration "
                    f"({', '.join(differing)}); a grid is not resumed across configurations, "
                    "choose another --out"
                )
            return True
        return False

    def require_configuration(self, *, task: str, commit: str) -> None:
        """Refuse to continue a directory whose stored runs belong to another task or commit.

        Checked once before any cell trains, so a session that would have to be refused is
        refused before it spends an hour rather than at the first cell it meets again.

        Raises:
            SystemExit: If a stored run names another task or another commit.
        """
        found = sorted({(row.get("task") or "", row["commit"]) for row in self.runs()})
        other = [f"{t or '?'} at {c[:7]}" for t, c in found if (t, c) != (task, commit)]
        if other:
            raise SystemExit(
                f"{self.directory} holds runs of {', '.join(other)}, not of {task} at "
                f"{commit[:7]}; a grid is not resumed across tasks or commits, choose another --out"
            )

    def cells(self) -> set[tuple[str, str, str]]:
        """The keys of the cells the directory holds whole."""
        return {(row["mode"], row["budget"], row["sample_seed"]) for row in self.runs()}

    def add(self, outcome: AdaptationOutcome, *, task: str, device: str, commit: str) -> None:
        """Append the cell's predictions, its epochs and, last, its run to the three files."""
        budget = budget_text(outcome.budget)
        cell = {"mode": str(outcome.plan.mode), "budget": budget, "seed": outcome.sample_seed}
        self._append(
            PREDICTIONS,
            (*CELL_COLUMNS, "unit", "position", "ends_at", "target", "predicted"),
            [
                {
                    **cell,
                    "unit": str(p.window.unit),
                    "position": p.window.position,
                    "ends_at": p.window.ends_at,
                    "target": p.target,
                    "predicted": f"{p.predicted:.6f}",
                }
                for p in outcome.predictions
            ],
        )
        self._append(
            EPOCHS,
            (*CELL_COLUMNS, "epoch", "loss"),
            [
                {**cell, "epoch": epoch, "loss": f"{loss:.8f}"}
                for epoch, loss in enumerate(outcome.training_losses)
            ],
        )
        self._append(
            RUNS,
            RUN_COLUMNS,
            [
                {
                    **outcome.plan.parameters(),
                    "task": task,
                    "budget": budget,
                    "windows": outcome.labelled_windows,
                    "steps": outcome.optimiser_steps,
                    "sample_seed": outcome.sample_seed,
                    "engines": outcome.labelled_units,
                    "trainable_parameters": outcome.trainable_parameters,
                    "rmse": f"{outcome.rmse:.6f}",
                    "seconds": f"{outcome.seconds:.3f}",
                    "device": device,
                    "commit": commit,
                    "torch": version("torch"),
                }
            ],
        )

    def runs(self) -> list[dict[str, str]]:
        return self._rows(RUNS)

    def predictions(self) -> Iterable[dict[str, str]]:
        """The stored predictions of the cells held whole, row by row."""
        committed = self.cells()
        for row in self._rows(PREDICTIONS):
            if (row["mode"], row["budget"], row["seed"]) in committed:
                yield row

    def publish(self, store: ArtifactStore) -> ArtifactRef:
        """Store the three files as one archive in the bucket; the reference resolves to it."""
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / f"{self.directory.name}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name in (RUNS, EPOCHS, PREDICTIONS):
                    if (self.directory / name).is_file():
                        tar.add(self.directory / name, arcname=name)
            return store.put_file(archive)

    @classmethod
    def fetch(cls, store: ArtifactStore, ref: ArtifactRef, directory: Path) -> Self:
        """Unpack a published report into ``directory``.

        Raises:
            SystemExit: If a report is already stored there.
        """
        if (directory / RUNS).exists():
            raise SystemExit(f"{directory} already holds {RUNS}; choose another --out")
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "report.tar.gz"
            store.get_file(ref, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(directory, filter="data")
        return cls(directory=directory)

    def _rows(self, name: str) -> list[dict[str, str]]:
        path = self.directory / name
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _append(
        self, name: str, columns: Sequence[str], rows: Sequence[Mapping[str, object]]
    ) -> None:
        path = self.directory / name
        new = not path.exists()
        if not new:
            with path.open(newline="", encoding="utf-8") as handle:
                header = tuple(csv.DictReader(handle).fieldnames or ())
            if header != tuple(columns):
                raise SystemExit(
                    f"{path} was written under other columns ({', '.join(header)}); a grid is "
                    "not continued across versions of the report, choose another --out"
                )
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerows(rows)

    def _discard_uncommitted(self) -> None:
        committed = self.cells()
        for name in (EPOCHS, PREDICTIONS):
            path = self.directory / name
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or ()
                rows = list(reader)
            kept = [row for row in rows if (row["mode"], row["budget"], row["seed"]) in committed]
            if len(kept) == len(rows):
                continue
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(kept)
