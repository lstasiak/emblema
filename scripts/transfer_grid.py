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
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef
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
class KnownTask:
    """What a supervised task is made of, as far as no adapter can read it off the corpus.

    Attributes:
        name: What the task is called in the reports.
        corpus: Name the corpus was published under.
        unit_prefix: What the keys of the task's units start with, inside that corpus.
        ceiling: Largest remaining life a window is labelled with.
        strata: How many groups of the target a budget is spread over.
        test_source: Where the frozen test engines come from, as the set is known outside.
        test_units: Keys of the frozen test engines.
    """

    name: str
    corpus: str
    unit_prefix: str
    ceiling: float
    strata: int
    test_source: str
    test_units: tuple[str, ...]

    def units_of(self, keys: Sequence[str]) -> frozenset[UnitKey]:
        """The task's units among the keys a published corpus names."""
        return frozenset(UnitKey(key) for key in keys if key.startswith(self.unit_prefix))


class KnownTasks:
    """The tasks a report may run, each stated once."""

    TURBOFAN_FD001 = KnownTask(
        name="turbofan-fd001",
        corpus="cmapss",
        unit_prefix="FD001/",
        ceiling=125.0,
        strata=4,
        test_source="cmapss/test/FD001",
        test_units=tuple(f"FD001/test/{engine}" for engine in range(1, 101)),
    )

    @classmethod
    def default(cls) -> KnownTask:
        return cls.TURBOFAN_FD001


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

    def holds(self, cell: Cell, plan: AdaptationPlan, *, commit: str) -> bool:
        """Whether the cell is stored under this plan and this commit.

        Raises:
            SystemExit: If the cell is stored under another plan or commit: a grid resumed
                across configurations would mix cells that are not comparable.
        """
        stated = {name: str(value) for name, value in plan.parameters().items()}
        for row in self.runs():
            if (row["mode"], row["budget"], row["sample_seed"]) != cell.key:
                continue
            differing = [name for name, value in stated.items() if row.get(name) != value]
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

    def cells(self) -> set[tuple[str, str, str]]:
        """The keys of the cells the directory holds whole."""
        return {(row["mode"], row["budget"], row["sample_seed"]) for row in self.runs()}

    def add(self, outcome: AdaptationOutcome, *, device: str, commit: str) -> None:
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
