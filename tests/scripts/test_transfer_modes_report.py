"""What the transfer report composes, stores and skips; no backbone is adapted here."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.evaluation.adapters.synthetic.synthetic_ground_truth import SyntheticGroundTruth
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from scripts.transfer_grid import Cell, Stored
from scripts.transfer_modes_report import cells_of, ground_truth_of, parse, plans_of, render
from tests.scripts.test_transfer_grid import outcome

pytestmark = pytest.mark.ml

WEIGHTS = ("durable/sha256/" + "a" * 64, "sha256:" + "a" * 64)
MANIFEST = ("durable/sha256/" + "b" * 64, "sha256:" + "b" * 64)


def arguments(*extra: str) -> list[str]:
    return ["--weights", *WEIGHTS, "--manifest", *MANIFEST, *extra]


def test_every_mode_gets_a_plan_with_its_own_learning_rate_under_one_shape() -> None:
    plans = plans_of(
        parse(
            arguments(
                "--lr-lora",
                "5e-3",
                "--lora-rank",
                "4",
                "--epochs",
                "3",
                "--final-lr-fraction",
                "0.01",
            )
        ),
        seed=4,
    )

    assert set(plans) == set(TransferMode)
    lora, frozen = plans[TransferMode.LORA], plans[TransferMode.FROZEN_PROBE]
    assert lora.schedule.learning_rate == 5e-3
    assert lora.lora is not None
    assert lora.lora.rank == 4
    assert plans[TransferMode.FULL_FINE_TUNING].schedule.learning_rate == 3e-4
    assert plans[TransferMode.FROM_SCRATCH].backbone is None
    assert frozen.backbone is not None
    assert frozen.backbone.key == WEIGHTS[0]
    assert all(stated.schedule.epochs == 3 for stated in plans.values())
    assert all(stated.schedule.warmup_fraction == 0.1 for stated in plans.values())
    assert all(stated.schedule.final_lr_fraction == 0.01 for stated in plans.values())
    assert plans[TransferMode.FROM_SCRATCH].schedule.learning_rate == 3e-4
    assert all(stated.seed == 4 for stated in plans.values())


def test_only_the_modes_asked_for_are_planned() -> None:
    plans = plans_of(parse(arguments("--mode", "lora", "--mode", "from_scratch")), seed=1)

    assert set(plans) == {TransferMode.LORA, TransferMode.FROM_SCRATCH}


def test_the_grid_is_every_mode_at_every_budget_under_every_seed_cheapest_first() -> None:
    cells = [cell for cell, _ in cells_of(parse(arguments()))]

    assert len(cells) == 4 * 4 * 5
    assert cells[0] == Cell(mode=TransferMode.FROM_SCRATCH, budget=LabelBudget.of(50), seed=1)
    assert str(cells[0]) == "from_scratch at 50 under seed 1"
    assert [cell.seed for cell in cells[:16]] == [1] * 16
    assert [str(cell.budget.windows) for cell in cells[:16:4]] == ["50", "200", "1000", "None"]
    assert cells[-1] == Cell(
        mode=TransferMode.FULL_FINE_TUNING, budget=LabelBudget.everything(), seed=5
    )


def test_a_shard_names_its_own_seeds_and_budgets() -> None:
    cells = [
        cell
        for cell, plan in cells_of(
            parse(arguments("--seed", "2", "--seed", "4", "--budget", "200", "--mode", "lora"))
        )
        if plan.seed == cell.seed
    ]

    assert [(cell.seed, cell.budget) for cell in cells] == [
        (2, LabelBudget.of(200)),
        (4, LabelBudget.of(200)),
    ]


def test_stored_cells_render_into_the_notes_table(tmp_path: Path) -> None:
    stored = Stored.open(tmp_path / "run")
    stored.add(
        outcome(TransferMode.FROZEN_PROBE), task="turbofan-fd001", device="cpu", commit="abc123"
    )
    stored.add(outcome(TransferMode.LORA), task="turbofan-fd001", device="cpu", commit="abc123")

    rendered = render(stored, KnownTasks.default())

    assert "| lr | warm-up | final lr |" in rendered
    assert (
        "| frozen_probe | 50 | 50 | 2 | 1 | 2 | 50 | 0.01 | 0.0 | 1.0 | 33 | 3.54 | 2 | cpu |"
        in rendered
    )
    assert "| lora |" in rendered
    assert "validation, not test" in rendered


def test_a_run_stored_before_a_column_existed_still_renders(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "runs.csv").write_text(
        "mode,backbone,run_seed,epochs,batch_size,learning_rate,weight_decay,lora_rank,"
        "lora_alpha,lora_dropout,lora_targets,budget,sample_seed,trainable_parameters,rmse,"
        "seconds,device,commit,torch\n"
        "lora,durable/w,1,30,16,0.001,0.0,8,16.0,0.0,qkv,200,1,196865,24.07,232.0,mps,abc,2.14\n"
    )

    rendered = render(Stored.existing(tmp_path / "old"), KnownTasks.default())

    assert "| lora | 200 |  |  | 1 | 30 |  | 0.001 |  |  | 196865 | 24.07 | 232 | mps |" in rendered


def test_a_synthetic_task_reads_its_truth_off_the_generator() -> None:
    assert isinstance(ground_truth_of(KnownTasks.CONTROL_B_FORECAST), SyntheticGroundTruth)
    assert isinstance(ground_truth_of(KnownTasks.NULL_B_FORECAST), SyntheticGroundTruth)


def test_a_real_task_whose_corpus_was_not_fetched_stops_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.transfer_modes_report.raw_root", lambda corpus: None)

    with pytest.raises(SystemExit, match="not under data/raw"):
        ground_truth_of(KnownTasks.default())


def test_the_task_is_an_argument_and_defaults_to_the_turbofan_one() -> None:
    assert parse(arguments()).task == "turbofan-fd001"
    assert parse(arguments("--task", "null-b-forecast")).task == "null-b-forecast"
