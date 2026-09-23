"""How a grid is stored, resumed and read back; nothing is trained and torch is not imported."""

from pathlib import Path

import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from scripts.transfer_grid import (
    EPOCHS,
    PLAN_COLUMNS,
    PREDICTIONS,
    RUNS,
    Cell,
    HeldOutShare,
    KnownTasks,
    NamedTestUnits,
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
        artifact=None,
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
    assert isinstance(task.test, NamedTestUnits)
    assert len(task.test.units) == 100
    assert (task.labels, task.strata) == (RemainingLifeScheme(125.0), 4)
    assert task.labels_text == "remaining life under a ceiling of 125"


def test_the_synthetic_tasks_forecast_the_first_sensor_of_the_second_layout() -> None:
    for task in (KnownTasks.CONTROL_B_FORECAST, KnownTasks.NULL_B_FORECAST):
        assert task.labels == ForecastScheme("s01", 12.0)
        assert task.unit_prefix == f"{task.corpus}/"
        assert task.units_called == "units"
        assert task.labels_text == "the exact reading of s01 12 time units past the window"
    assert KnownTasks.named("null-b-forecast") is KnownTasks.NULL_B_FORECAST
    assert KnownTasks.names() == (
        "turbofan-fd001",
        "control-b-forecast",
        "null-b-forecast",
        "control-b-wide-forecast",
        "null-b-wide-forecast",
        "control-b-shared-forecast",
    )


def test_a_task_nobody_stated_is_refused() -> None:
    with pytest.raises(KeyError):
        KnownTasks.named("turbofan-fd009")


def test_a_named_test_side_is_the_units_it_names_whatever_is_held_out() -> None:
    frozen = KnownTasks.default().frozen_test(frozenset({UnitKey("FD001/7")}))

    assert len(frozen.units) == 100
    assert UnitKey("FD001/7") not in frozen.units
    assert frozen.source == "cmapss/test/FD001"


def test_a_held_out_share_freezes_the_same_third_however_the_units_are_named() -> None:
    held_out = frozenset(UnitKey(f"control-b/{index}") for index in range(30))
    task = KnownTasks.CONTROL_B_FORECAST

    first, again = (
        task.frozen_test(held_out),
        task.frozen_test(frozenset(sorted(held_out, key=str))),
    )

    assert first == again
    assert len(first.units) == 10
    assert first.units < held_out
    assert first.source == "control-b/held-out"
    assert first.units != frozenset(sorted(held_out, key=str)[:10])


def test_a_plan_flattens_to_the_columns_a_run_is_stored_under() -> None:
    assert tuple(plan().parameters()) == PLAN_COLUMNS


def test_a_stored_cell_is_three_files_keyed_by_the_cell(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(
        outcome(TransferMode.FROZEN_PROBE), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    stored.add(
        outcome(TransferMode.LORA, seed=3), task="turbofan-fd001", device="cpu", commit="abc123"
    )

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
    stored.add(
        outcome(TransferMode.LORA, seed=3), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    reopened = Stored.open(tmp_path / "run")
    stated = plan(TransferMode.LORA, seed=3)

    assert reopened.holds(cell(seed=3), stated, task="turbofan-fd001", commit="abc123")
    assert not reopened.holds(
        cell(seed=1), plan(TransferMode.LORA, seed=1), task="turbofan-fd001", commit="abc123"
    )
    assert not reopened.holds(
        cell(budget=200, seed=3), stated, task="turbofan-fd001", commit="abc123"
    )
    assert not Stored.open(tmp_path / "empty").holds(
        cell(), stated, task="turbofan-fd001", commit="abc123"
    )


def test_a_cell_stored_under_another_plan_or_commit_is_not_resumed_over(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(
        outcome(TransferMode.LORA, seed=3), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    stated = plan(TransferMode.LORA, seed=3)

    with pytest.raises(SystemExit, match=r"another configuration \(commit\)"):
        stored.holds(cell(seed=3), stated, task="turbofan-fd001", commit="def456")
    other = plan(
        TransferMode.LORA,
        seed=3,
        schedule=adaptation_schedule(learning_rate=5e-3, warmup_fraction=0.1),
    )
    with pytest.raises(SystemExit, match=r"\(learning_rate, warmup_fraction\)"):
        stored.holds(cell(seed=3), other, task="turbofan-fd001", commit="abc123")


def test_a_cell_interrupted_before_its_run_row_is_dropped_when_the_directory_is_reopened(
    tmp_path: Path,
) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(
        outcome(TransferMode.LORA, seed=1), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    # A session killed between the predictions and the run row leaves the former without the
    # latter; the cell is not held, and what it left must not double the rerun's predictions.
    with (tmp_path / "run" / PREDICTIONS).open("a") as handle:
        handle.write("lora,50,2,a,0,0.0,10.0,11.0\nlora,50,2,b,1,1.0,20.0,21.0\n")
    with (tmp_path / "run" / EPOCHS).open("a") as handle:
        handle.write("lora,50,2,0,0.5\n")

    assert not Stored.existing(tmp_path / "run").holds(
        cell(seed=2), plan(TransferMode.LORA, seed=2), task="turbofan-fd001", commit="abc123"
    )
    assert [row["seed"] for row in Stored.existing(tmp_path / "run").predictions()] == ["1", "1"]
    reopened = Stored.open(tmp_path / "run")
    assert len((tmp_path / "run" / PREDICTIONS).read_text().splitlines()) == 1 + 2
    assert len((tmp_path / "run" / EPOCHS).read_text().splitlines()) == 1 + 2
    reopened.add(
        outcome(TransferMode.LORA, seed=2), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    assert [row["seed"] for row in reopened.predictions()] == ["1", "1", "2", "2"]


def test_a_published_report_fetches_back_file_for_file(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA), task="turbofan-fd001", device="cpu", commit="abc123")

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


def test_a_share_of_fewer_than_one_in_two_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        HeldOutShare(source="x/held-out", one_in=1, seed=1)


def test_a_held_out_share_rounds_up_so_a_small_corpus_still_freezes_a_unit() -> None:
    share = HeldOutShare(source="x/held-out", one_in=3, seed=1)

    assert len(share.frozen_split(frozenset({UnitKey("x/0")})).units) == 1
    assert len(share.frozen_split(frozenset(UnitKey(f"x/{i}") for i in range(4))).units) == 2


def test_a_directory_of_another_task_or_commit_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(outcome(TransferMode.LORA), task="turbofan-fd001", device="cpu", commit="abc123")

    stored.require_configuration(task="turbofan-fd001", commit="abc123")
    with pytest.raises(SystemExit, match="not resumed across tasks or commits"):
        stored.require_configuration(task="turbofan-fd001", commit="def456")
    with pytest.raises(SystemExit, match="not resumed across tasks or commits"):
        stored.require_configuration(task="null-b-forecast", commit="abc123")
    Stored.open(tmp_path / "empty").require_configuration(task="null-b-forecast", commit="x")


def test_a_file_written_under_other_columns_is_not_appended_to(tmp_path: Path) -> None:
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / RUNS).write_text("mode,budget,seed,rmse\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="other columns"):
        Stored.open(directory).add(
            outcome(TransferMode.LORA), task="turbofan-fd001", device="cpu", commit="abc123"
        )
