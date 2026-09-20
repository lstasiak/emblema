"""How a grid is stored, resumed and read back; nothing is trained and torch is not imported."""

from pathlib import Path

import pytest

from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from scripts.transfer_grid import (
    EPOCHS,
    PLAN_COLUMNS,
    PREDICTIONS,
    RUNS,
    Cell,
    KnownTasks,
    Stored,
    budget_of,
    budget_rank,
)
from tests.evaluation.support import TASK, adaptation_schedule, plan, prediction


def outcome(mode: TransferMode, seed: int = 1) -> AdaptationOutcome:
    return AdaptationOutcome(
        plan=plan(mode, seed=seed),
        task=TASK,
        budget=LabelBudget.of(50),
        sample_seed=seed,
        labelled_windows=50,
        labelled_units=2,
        trainable_parameters=33,
        training_losses=(0.5, 0.25),
        predictions=(prediction("a", 0, 10.0, 13.0), prediction("b", 1, 20.0, 16.0)),
        seconds=2.0,
    )


def cell(mode: TransferMode = TransferMode.LORA, budget: int = 50, seed: int = 1) -> Cell:
    return Cell(mode=mode, budget=LabelBudget.of(budget), seed=seed)


def test_a_budget_is_a_count_of_windows_or_everything_and_sorts_by_its_count() -> None:
    assert budget_of("200") == LabelBudget.of(200)
    assert budget_of("all") == LabelBudget.everything()
    assert sorted(["all", "500", "50", "1000", "100", "200"], key=budget_rank) == [
        "50",
        "100",
        "200",
        "500",
        "1000",
        "all",
    ]


def test_the_turbofan_task_is_cut_from_the_engines_of_its_subset() -> None:
    task = KnownTasks.default()

    units = task.units_of(["FD001/1", "FD002/1", "FD001/100", "FD003/7"])

    assert {str(unit) for unit in units} == {"FD001/1", "FD001/100"}
    assert len(task.test_units) == 100
    assert (task.ceiling, task.strata) == (125.0, 4)


def test_a_plan_flattens_to_the_columns_a_run_is_stored_under() -> None:
    assert tuple(plan().parameters()) == PLAN_COLUMNS


def test_a_stored_cell_is_three_files_keyed_by_the_cell(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.FROZEN_PROBE), device="cpu", commit="abc123")
    stored.add(outcome(TransferMode.LORA, seed=3), device="cpu", commit="abc123")

    (first, second) = stored.runs()

    assert (first["run_seed"], first["sample_seed"], first["engines"]) == ("1", "1", "2")
    assert "seed" not in first
    assert (second["mode"], second["sample_seed"]) == ("lora", "3")
    assert len((tmp_path / "run" / EPOCHS).read_text().splitlines()) == 1 + 2 * 2
    assert len((tmp_path / "run" / PREDICTIONS).read_text().splitlines()) == 1 + 2 * 2
    assert len((tmp_path / "run" / RUNS).read_text().splitlines()) == 1 + 2
    epochs = (tmp_path / "run" / EPOCHS).read_text().splitlines()
    assert epochs[0] == "mode,budget,seed,epoch,loss"
    assert epochs[3].startswith("lora,50,3,0,")
    predictions = (tmp_path / "run" / PREDICTIONS).read_text().splitlines()
    assert predictions[0] == "mode,budget,seed,unit,position,ends_at,target,predicted"
    assert stored.cells() == {("frozen_probe", "50", "1"), ("lora", "50", "3")}


def test_a_cell_stored_under_the_same_plan_and_commit_is_held(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA, seed=3), device="cpu", commit="abc123")
    reopened = Stored.open(tmp_path / "run")
    stated = plan(TransferMode.LORA, seed=3)

    assert reopened.holds(cell(seed=3), stated, commit="abc123")
    assert not reopened.holds(cell(seed=1), plan(TransferMode.LORA, seed=1), commit="abc123")
    assert not reopened.holds(cell(budget=200, seed=3), stated, commit="abc123")
    assert not Stored.open(tmp_path / "empty").holds(cell(), stated, commit="abc123")


def test_a_cell_stored_under_another_plan_or_commit_is_not_resumed_over(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA, seed=3), device="cpu", commit="abc123")
    stated = plan(TransferMode.LORA, seed=3)

    with pytest.raises(SystemExit, match=r"another configuration \(commit\)"):
        stored.holds(cell(seed=3), stated, commit="def456")
    other = plan(
        TransferMode.LORA,
        seed=3,
        schedule=adaptation_schedule(learning_rate=5e-3, warmup_fraction=0.1),
    )
    with pytest.raises(SystemExit, match=r"\(learning_rate, warmup_fraction\)"):
        stored.holds(cell(seed=3), other, commit="abc123")


def test_a_cell_interrupted_before_its_run_row_is_dropped_when_the_directory_is_reopened(
    tmp_path: Path,
) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA, seed=1), device="cpu", commit="abc123")
    # A session killed between the predictions and the run row leaves the former without the
    # latter; the cell is not held, and what it left must not double the rerun's predictions.
    with (tmp_path / "run" / PREDICTIONS).open("a") as handle:
        handle.write("lora,50,2,a,0,0.0,10.0,11.0\nlora,50,2,b,1,1.0,20.0,21.0\n")
    with (tmp_path / "run" / EPOCHS).open("a") as handle:
        handle.write("lora,50,2,0,0.5\n")

    assert not Stored.existing(tmp_path / "run").holds(
        cell(seed=2), plan(TransferMode.LORA, seed=2), commit="abc123"
    )
    assert [row["seed"] for row in Stored.existing(tmp_path / "run").predictions()] == ["1", "1"]
    reopened = Stored.open(tmp_path / "run")
    assert len((tmp_path / "run" / PREDICTIONS).read_text().splitlines()) == 1 + 2
    assert len((tmp_path / "run" / EPOCHS).read_text().splitlines()) == 1 + 2
    reopened.add(outcome(TransferMode.LORA, seed=2), device="cpu", commit="abc123")
    assert [row["seed"] for row in reopened.predictions()] == ["1", "1", "2", "2"]


def test_a_published_report_fetches_back_file_for_file(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA), device="cpu", commit="abc123")

    ref = stored.publish(store)
    fetched = Stored.fetch(store, ref, tmp_path / "fetched")

    for name in (RUNS, EPOCHS, PREDICTIONS):
        assert (tmp_path / "fetched" / name).read_bytes() == (tmp_path / "run" / name).read_bytes()
    assert fetched.runs() == stored.runs()
    with pytest.raises(SystemExit, match="already holds"):
        Stored.fetch(store, ref, tmp_path / "fetched")


def test_rendering_a_directory_that_holds_no_report_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nothing to render"):
        Stored.existing(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()
