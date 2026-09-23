"""Which of two backbones the replacement rule keeps, read from stored cells; nothing is trained."""

import random
from pathlib import Path

import pytest

from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.backbone_comparison_report import compare, main, side_of
from scripts.transfer_grid import Stored
from tests.evaluation.support import TASK, plan, prediction

OLD = ArtifactRef(key="durable/old", checksum=Checksum.of_bytes(b"old"))
NEW = ArtifactRef(key="durable/new", checksum=Checksum.of_bytes(b"new"))
UNITS = ("a", "b", "c", "d", "e", "f")
WINDOWS = 4


def outcome(
    backbone: ArtifactRef | None,
    seed: int,
    error: float,
    *,
    mode: TransferMode = TransferMode.FULL_FINE_TUNING,
    shift: float = 0.0,
) -> AdaptationOutcome:
    draws = random.Random(seed)
    return AdaptationOutcome(
        plan=plan(mode, seed=seed, backbone=backbone),
        task=TASK,
        budget=LabelBudget.of(200),
        sample_seed=seed,
        labelled_windows=200,
        labelled_units=4,
        trainable_parameters=33,
        training_losses=(0.5, 0.25),
        predictions=tuple(
            prediction(
                unit,
                index * WINDOWS + place,
                100.0 - 10.0 * place + shift,
                (100.0 - 10.0 * place) * (1.0 + error * draws.uniform(0.6, 1.4)),
            )
            for index, unit in enumerate(UNITS)
            for place in range(WINDOWS)
        ),
        seconds=3.0,
        artifact=None,
    )


def shard(directory: Path, *outcomes: AdaptationOutcome) -> Path:
    stored = Stored.open(directory)
    for each in outcomes:
        stored.add(each, task="turbofan-fd001", device="cpu", commit="abc")
    return directory


def test_a_backbone_whose_arm_errs_less_on_every_engine_replaces_the_old(tmp_path: Path) -> None:
    old = side_of(
        [shard(tmp_path / "old", *(outcome(OLD, s, 0.3) for s in (1, 2)))],
        "full_fine_tuning",
        "200",
    )
    new = side_of(
        [shard(tmp_path / "new", *(outcome(NEW, s, 0.1) for s in (1, 2)))],
        "full_fine_tuning",
        "200",
    )

    comparison = compare(old, new, mode="full_fine_tuning", budget="200", resamples=500)

    assert comparison.seeds == (1, 2)
    assert comparison.engines == len(UNITS)
    assert comparison.rmse_new < comparison.rmse_old
    assert comparison.replaces
    assert comparison.render().endswith("the new backbone replaces the old")


def test_the_old_backbone_stays_when_the_new_one_is_worse(tmp_path: Path) -> None:
    old = side_of([shard(tmp_path / "old", outcome(OLD, 1, 0.1))], "full_fine_tuning", "200")
    new = side_of([shard(tmp_path / "new", outcome(NEW, 1, 0.3))], "full_fine_tuning", "200")

    comparison = compare(old, new, mode="full_fine_tuning", budget="200", resamples=500)

    assert not comparison.replaces
    assert comparison.render().endswith("the old backbone stays")


def test_only_the_seeds_both_sides_hold_are_paired(tmp_path: Path) -> None:
    old = side_of(
        [shard(tmp_path / "old", *(outcome(OLD, s, 0.3) for s in (1, 2, 3)))],
        "full_fine_tuning",
        "200",
    )
    new = side_of(
        [shard(tmp_path / "new", *(outcome(NEW, s, 0.1) for s in (2, 3, 4)))],
        "full_fine_tuning",
        "200",
    )

    assert compare(old, new, mode="full_fine_tuning", budget="200", resamples=10).seeds == (2, 3)


def test_the_mode_compared_is_read_apart_from_the_other_modes_in_a_directory(
    tmp_path: Path,
) -> None:
    directory = shard(
        tmp_path / "mixed",
        outcome(NEW, 1, 0.1),
        outcome(NEW, 1, 0.2, mode=TransferMode.LORA),
        outcome(None, 1, 0.3, mode=TransferMode.FROM_SCRATCH),
    )

    side = side_of([directory], "full_fine_tuning", "200")

    assert side.backbone == NEW.key
    assert list(side.cells) == [1]


def test_a_pair_that_cannot_be_read_as_one_is_refused(tmp_path: Path) -> None:
    old = side_of([shard(tmp_path / "old", outcome(OLD, 1, 0.3))], "full_fine_tuning", "200")
    same = side_of([shard(tmp_path / "same", outcome(OLD, 1, 0.1))], "full_fine_tuning", "200")
    elsewhere = side_of(
        [shard(tmp_path / "elsewhere", outcome(NEW, 2, 0.1))], "full_fine_tuning", "200"
    )
    moved = side_of(
        [shard(tmp_path / "moved", outcome(NEW, 1, 0.1, shift=1.0))], "full_fine_tuning", "200"
    )

    with pytest.raises(SystemExit, match="the same backbone"):
        compare(old, same, mode="full_fine_tuning", budget="200")
    with pytest.raises(SystemExit, match="no seed is held by both sides"):
        compare(old, elsewhere, mode="full_fine_tuning", budget="200")
    with pytest.raises(SystemExit, match="different windows or targets"):
        compare(old, moved, mode="full_fine_tuning", budget="200")


def test_a_side_under_two_backbones_or_none_is_refused(tmp_path: Path) -> None:
    first = shard(tmp_path / "first", outcome(OLD, 1, 0.3))
    second = shard(tmp_path / "second", outcome(NEW, 2, 0.3))

    with pytest.raises(SystemExit, match="several backbones"):
        side_of([first, second], "full_fine_tuning", "200")
    with pytest.raises(SystemExit, match="no cell of lora at 200"):
        side_of([first], "lora", "200")


def test_the_verdict_is_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old = shard(tmp_path / "old", outcome(OLD, 1, 0.3))
    new = shard(tmp_path / "new", outcome(NEW, 1, 0.1))

    main(["--old", str(old), "--new", str(new)])

    printed = capsys.readouterr().out
    assert printed.startswith("full_fine_tuning at 200, seeds [1]")
    assert "the new backbone replaces the old" in printed
