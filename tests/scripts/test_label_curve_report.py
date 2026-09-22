"""What the curve report composes from stored cells and what it concludes; nothing is trained."""

import random
from pathlib import Path

import pytest

from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from scripts.label_curve_report import (
    BASELINES,
    COMPARISONS,
    CURVE,
    curve_of,
    read,
    render,
    sentence,
    write,
)
from scripts.transfer_grid import KnownTasks, Stored
from tests.evaluation.support import TASK, adaptation_schedule, plan, prediction

UNITS = ("a", "b", "c", "d", "e")
WINDOWS = 4
BUDGETS = ("50", "200")
SEEDS = (1, 2)
INCOMPLETE = (
    "Incomplete grid — 6 of 12 cells compared, 2 of 5 seeds pooled — so not the registered "
    "reading. "
)
# The error each mode makes, as a share of the label: the control far off, the probe near it,
# the two arms that update the encoder far under it.
ERROR = {
    TransferMode.FROM_SCRATCH: 0.5,
    TransferMode.FROZEN_PROBE: 0.45,
    TransferMode.LORA: 0.2,
    TransferMode.FULL_FINE_TUNING: 0.15,
}


def outcome(
    mode: TransferMode, budget: str, seed: int, error: float, **overrides: object
) -> AdaptationOutcome:
    draws = random.Random(seed * 100 + (0 if budget == "all" else int(budget)))
    return AdaptationOutcome(
        plan=plan(mode, seed=seed, **overrides),
        task=TASK,
        budget=LabelBudget.everything() if budget == "all" else LabelBudget.of(int(budget)),
        sample_seed=seed,
        labelled_windows=2651 if budget == "all" else int(budget),
        labelled_units=3 if budget == "50" else 4,
        trainable_parameters=33,
        training_losses=(0.5, 0.25),
        predictions=tuple(
            prediction(
                unit,
                index * WINDOWS + place,
                100.0 - 10.0 * place,
                (100.0 - 10.0 * place) * (1.0 + error * draws.uniform(0.6, 1.4)),
            )
            for index, unit in enumerate(UNITS)
            for place in range(WINDOWS)
        ),
        seconds=3.0,
    )


@pytest.fixture
def shards(tmp_path: Path) -> list[Stored]:
    """Two shards, one per seed, every mode at every budget."""
    stored = []
    for seed in SEEDS:
        shard = Stored.open(tmp_path / f"shard-{seed}")
        for budget in BUDGETS:
            for mode in TransferMode:
                shard.add(
                    outcome(mode, budget, seed, ERROR[mode]),
                    task="turbofan-fd001",
                    device="cpu",
                    commit="abc",
                )
        stored.append(shard)
    return stored


def test_every_stored_run_becomes_a_point_with_its_readings(shards: list[Stored]) -> None:
    curve = curve_of(shards, KnownTasks.default())

    assert len(curve.points) == 4 * 2 * 2
    first = curve.points[0]
    assert (first.mode, first.budget, first.seed, first.engines) == ("from_scratch", "50", 1, 3)
    assert first.rmse > 0.0
    assert first.rmse_below_ceiling == pytest.approx(first.rmse)
    assert first.alpha_lambda_accuracy is not None
    assert 0.0 <= first.alpha_lambda_accuracy <= 1.0
    assert [b.name for b in curve.baselines] == ["mean predictor", "ceiling predictor"]
    assert curve.baselines[0].rmse == pytest.approx(11.180339887, abs=1e-6)


def test_every_mode_but_the_control_is_compared_at_every_budget(shards: list[Stored]) -> None:
    curve = curve_of(shards, KnownTasks.default())

    assert [(row.mode, row.budget) for row in curve.comparisons] == [
        ("frozen_probe", "50"),
        ("lora", "50"),
        ("full_fine_tuning", "50"),
        ("frozen_probe", "200"),
        ("lora", "200"),
        ("full_fine_tuning", "200"),
    ]
    assert all(row.seeds == 2 for row in curve.comparisons)
    primary = next(row for row in curve.comparisons if row.primary)
    assert (primary.mode, primary.budget) == ("full_fine_tuning", "200")
    assert primary.verdict == "confirmed"
    assert primary.rejected
    assert primary.relative_reduction > 0.10
    assert 0.0 < primary.low < primary.reduction < primary.high
    assert primary.floor >= 0.03 * primary.control_rmse


def test_a_secondary_cells_verdict_is_the_familys_word_and_the_family_is_the_registered_one(
    shards: list[Stored],
) -> None:
    curve = curve_of(shards, KnownTasks.default())

    for row in curve.comparisons:
        if row.primary:
            continue
        assert row.rejected == (row.verdict != "indistinguishable"), row
        assert row.verdict != "confirmed", row
    assert "the registered family of 11" in render(curve, KnownTasks.default())


def test_budgets_sort_by_their_count_of_windows_with_everything_last(tmp_path: Path) -> None:
    shard = Stored.open(tmp_path / "shard")
    for budget in ("all", "200", "100", "50", "1000"):
        for mode in (TransferMode.FROM_SCRATCH, TransferMode.LORA):
            shard.add(
                outcome(mode, budget, 1, ERROR[mode]),
                task="turbofan-fd001",
                device="cpu",
                commit="abc",
            )

    curve = curve_of([shard], KnownTasks.default())

    assert [p.budget for p in curve.points[::2]] == ["50", "100", "200", "1000", "all"]
    assert [row.budget for row in curve.comparisons] == ["50", "100", "200", "1000", "all"]


def test_shards_that_do_not_make_one_grid_are_refused(tmp_path: Path) -> None:
    first = Stored.open(tmp_path / "first")
    first.add(
        outcome(TransferMode.FROM_SCRATCH, "50", 1, 0.5),
        task="turbofan-fd001",
        device="cpu",
        commit="abc",
    )
    first.add(
        outcome(TransferMode.LORA, "50", 1, 0.2), task="turbofan-fd001", device="cpu", commit="abc"
    )
    other_commit = Stored.open(tmp_path / "other-commit")
    other_commit.add(
        outcome(TransferMode.LORA, "50", 2, 0.2), task="turbofan-fd001", device="cpu", commit="def"
    )
    other_plan = Stored.open(tmp_path / "other-plan")
    other_plan.add(
        outcome(
            TransferMode.LORA,
            "50",
            2,
            0.2,
            schedule=adaptation_schedule(epochs=2, learning_rate=3e-3),
        ),
        task="turbofan-fd001",
        device="cpu",
        commit="abc",
    )

    with pytest.raises(SystemExit, match="more than one commit: abc, def"):
        curve_of([first, other_commit], KnownTasks.default())
    with pytest.raises(
        SystemExit, match="lora was run under more than one plan; differing: learning_rate"
    ):
        curve_of([first, other_plan], KnownTasks.default())


def test_the_probe_near_the_control_is_not_a_confirmed_advantage(shards: list[Stored]) -> None:
    curve = curve_of(shards, KnownTasks.default())

    probe = next(row for row in curve.comparisons if row.mode == "frozen_probe")
    assert probe.verdict != "confirmed"
    assert probe.reduction < curve.comparisons[2].reduction


def test_the_curve_round_trips_through_its_files(shards: list[Stored], tmp_path: Path) -> None:
    curve = curve_of(shards, KnownTasks.default())

    write(curve, tmp_path / "curve")

    assert {p.name for p in (tmp_path / "curve").iterdir()} == {CURVE, COMPARISONS, BASELINES}
    assert read(tmp_path / "curve") == curve


def test_the_conclusion_names_the_endpoint_and_the_secondary_shape(shards: list[Stored]) -> None:
    curve = curve_of(shards, KnownTasks.default())

    conclusion = sentence(curve)

    assert conclusion.startswith(
        f"{INCOMPLETE}Confirmed on the registered endpoint: at 200 labelled windows"
    )
    assert "Holm correction and above the floor at each of them (50)." in conclusion
    assert conclusion.endswith("Preliminary; validation, not test.")
    rendered = render(curve, KnownTasks.default())
    assert "| 50 | 3 |" in rendered
    assert "validation, not test" in rendered
    assert "**Conclusion.**" in rendered


def test_a_cell_held_by_two_shards_is_refused(shards: list[Stored], tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="repeats a cell"):
        curve_of([shards[0], shards[0]], KnownTasks.default())


def test_a_curve_without_the_endpoint_says_so(tmp_path: Path) -> None:
    shard = Stored.open(tmp_path / "shard")
    for mode in (TransferMode.FROM_SCRATCH, TransferMode.LORA):
        shard.add(
            outcome(mode, "50", 1, ERROR[mode]), task="turbofan-fd001", device="cpu", commit="abc"
        )

    curve = curve_of([shard], KnownTasks.default())

    assert sentence(curve) == (
        "Incomplete grid — 1 of 12 cells compared, 1 of 5 seeds pooled — so not the registered "
        "reading. The registered endpoint was not measured."
    )
    assert [row.primary for row in curve.comparisons] == [False]


def test_the_whole_grid_is_read_without_the_incomplete_warning(tmp_path: Path) -> None:
    shard = Stored.open(tmp_path / "shard")
    for seed in (1, 2, 3, 4, 5):
        for budget in ("50", "200", "1000", "all"):
            for mode in TransferMode:
                shard.add(
                    outcome(mode, budget, seed, ERROR[mode]),
                    task="turbofan-fd001",
                    device="cpu",
                    commit="abc",
                )

    curve = curve_of([shard], KnownTasks.default())

    assert len(curve.comparisons) == 12
    assert sentence(curve).startswith("Confirmed on the registered endpoint")
    assert "Incomplete" not in sentence(curve)


def test_the_conclusion_names_the_budget_of_every_label_in_words(tmp_path: Path) -> None:
    shard = Stored.open(tmp_path / "shard")
    for seed in (1, 2, 3, 4, 5):
        for budget in ("50", "200", "1000", "all"):
            for mode in TransferMode:
                # Full fine-tuning draws level with the control once every label is used.
                matched = (mode, budget) == (TransferMode.FULL_FINE_TUNING, "all")
                error = ERROR[TransferMode.FROM_SCRATCH] if matched else ERROR[mode]
                shard.add(
                    outcome(mode, budget, seed, error),
                    task="turbofan-fd001",
                    device="cpu",
                    commit="abc",
                )

    conclusion = sentence(curve_of([shard], KnownTasks.default()))

    assert "above the floor at 50 and 1000, not at the full label set." in conclusion
    assert "not at all" not in conclusion


def test_a_directory_without_a_curve_is_not_rendered(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nothing to render"):
        read(tmp_path / "missing")


def forecast_shard(directory: Path) -> Stored:
    """A shard of the synthetic forecasting task: the control and full fine-tuning at 200."""
    shard = Stored.open(directory)
    for mode in (TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING):
        for seed in SEEDS:
            shard.add(
                outcome(mode, "200", seed, ERROR[mode]),
                task=KnownTasks.CONTROL_B_FORECAST.name,
                device="cpu",
                commit="abc",
            )
    return shard


def test_a_forecasting_task_reads_the_endpoint_alone_and_no_ceiling(tmp_path: Path) -> None:
    task = KnownTasks.CONTROL_B_FORECAST

    curve = curve_of([forecast_shard(tmp_path / "forecast")], task)

    assert [b.name for b in curve.baselines] == ["mean predictor"]
    assert all(p.rmse > 0 for p in curve.points)
    assert all(
        (p.rmse_below_ceiling, p.last_window_rmse, p.alpha_lambda_accuracy, p.asymmetric_score)
        == (None, None, None, None)
        for p in curve.points
    )
    write(curve, tmp_path / "out")
    assert read(tmp_path / "out") == curve
    rendered = render(curve, task)
    assert "the exact reading of s01 12 time units past the window" in rendered
    assert "| budget | units |" in rendered
    assert "Alpha-lambda" not in rendered


def test_shards_of_another_task_than_asked_are_refused(tmp_path: Path) -> None:
    shard = forecast_shard(tmp_path / "forecast")

    with pytest.raises(SystemExit, match="pass --task"):
        curve_of([shard], KnownTasks.default())
