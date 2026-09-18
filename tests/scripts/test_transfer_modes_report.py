"""What the transfer report composes and stores; no backbone is adapted here."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from scripts.transfer_modes_report import (
    EPOCHS,
    PREDICTIONS,
    RUNS,
    KnownTasks,
    Stored,
    budget_of,
    parse,
    plans_of,
    render,
)
from tests.evaluation.support import TASK, plan, prediction

pytestmark = pytest.mark.ml

WEIGHTS = ("durable/sha256/" + "a" * 64, "sha256:" + "a" * 64)
MANIFEST = ("durable/sha256/" + "b" * 64, "sha256:" + "b" * 64)


def arguments(*extra: str) -> list[str]:
    return ["--weights", *WEIGHTS, "--manifest", *MANIFEST, *extra]


def test_every_mode_gets_a_plan_with_its_own_learning_rate() -> None:
    plans = plans_of(parse(arguments("--lr-lora", "5e-3", "--lora-rank", "4", "--epochs", "3")))

    assert set(plans) == set(TransferMode)
    lora, frozen = plans[TransferMode.LORA], plans[TransferMode.FROZEN_PROBE]
    assert lora.schedule.learning_rate == 5e-3
    assert lora.lora is not None
    assert lora.lora.rank == 4
    assert plans[TransferMode.FULL_FINE_TUNING].schedule.learning_rate == 1e-4
    assert plans[TransferMode.FROM_SCRATCH].backbone is None
    assert frozen.backbone is not None
    assert frozen.backbone.key == WEIGHTS[0]
    assert all(stated.schedule.epochs == 3 for stated in plans.values())


def test_only_the_modes_asked_for_are_planned() -> None:
    plans = plans_of(parse(arguments("--mode", "lora", "--mode", "from_scratch")))

    assert set(plans) == {TransferMode.LORA, TransferMode.FROM_SCRATCH}


def test_a_budget_is_a_count_of_windows_or_everything() -> None:
    assert budget_of("200") == LabelBudget.of(200)
    assert budget_of("all") == LabelBudget.everything()


def test_the_turbofan_task_is_cut_from_the_engines_of_its_subset() -> None:
    task = KnownTasks.default()

    units = task.units_of(["FD001/1", "FD002/1", "FD001/100", "FD003/7"])

    assert {str(unit) for unit in units} == {"FD001/1", "FD001/100"}
    assert len(task.test_units) == 100
    assert (task.ceiling, task.strata) == (125.0, 4)


def outcome(mode: TransferMode) -> AdaptationOutcome:
    return AdaptationOutcome(
        plan=plan(mode),
        task=TASK,
        budget=LabelBudget.of(50),
        sample_seed=1,
        trainable_parameters=33,
        training_losses=(0.5, 0.25),
        predictions=(prediction("a", 0, 10.0, 13.0), prediction("b", 1, 20.0, 16.0)),
        seconds=2.0,
    )


def test_stored_runs_render_into_the_notes_table(tmp_path: Path) -> None:
    stored = Stored.at(tmp_path / "run")
    stored.add(outcome(TransferMode.FROZEN_PROBE), device="cpu", commit="abc123")
    stored.add(outcome(TransferMode.LORA), device="cpu", commit="abc123")

    rendered = render(stored, KnownTasks.default())

    assert "| frozen_probe | 50 | 1 | 2 | 0.01 | 33 | 3.54 | 2 | cpu |" in rendered
    assert "| lora |" in rendered
    assert "validation, not test" in rendered
    assert len((tmp_path / "run" / EPOCHS).read_text().splitlines()) == 1 + 2 * 2
    assert len((tmp_path / "run" / PREDICTIONS).read_text().splitlines()) == 1 + 2 * 2
    assert len((tmp_path / "run" / RUNS).read_text().splitlines()) == 1 + 2


def test_the_two_seeds_of_a_run_are_named_apart_in_the_stored_row(tmp_path: Path) -> None:
    stored = Stored.at(tmp_path / "run")
    stored.add(outcome(TransferMode.FROZEN_PROBE), device="cpu", commit="abc123")

    (row,) = stored.runs()

    assert (row["run_seed"], row["sample_seed"]) == ("1", "1")
    assert "seed" not in row


def test_a_directory_that_already_holds_a_report_is_not_written_into(tmp_path: Path) -> None:
    Stored.at(tmp_path / "run").add(outcome(TransferMode.LORA), device="cpu", commit="abc123")

    with pytest.raises(SystemExit, match="already holds"):
        Stored.at(tmp_path / "run")


def test_rendering_a_directory_that_holds_no_report_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nothing to render"):
        Stored.existing(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()
