"""The pilot's cut, draw, loop and reading on small data; no backbone and no registry."""

import csv
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from emblema.evaluation.domain.exceptions import DivergedAdaptationError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from scripts.taught_fine_tuning_pilot import (
    ANSWER_COLUMNS,
    CELL_COLUMNS,
    BalancedDraw,
    corrected,
    cut_by_month,
    fine_tuned,
    gain_row,
    main,
    moved_share,
    read_answers,
    write_answers,
)

SCHEDULE = AdaptationSchedule(
    epochs=1,
    min_steps=40,
    batch_size=8,
    learning_rate=0.05,
    weight_decay=0.0,
    warmup_fraction=0.1,
    final_lr_fraction=0.1,
)
QUICK = PairedUnitBootstrap(resamples=50, seed=1)


def labelled(unit: str, position: int, target: float) -> LabelledWindow:
    return LabelledWindow(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
        target=target,
    )


def test_a_cut_keeps_the_months_apart_and_sends_a_straddling_event_to_neither() -> None:
    windows = [
        labelled("m/2000-01", 0, 0.0),
        labelled("m/2000-01", 1, 1.0),
        labelled("m/2000-02", 2, 2.0),
        labelled("m/2001-01", 3, 2.0),
        labelled("m/2001-01", 4, 3.0),
        labelled("m/2001-01", 5, 0.0),
        labelled("m/2009-01", 6, 4.0),
    ]

    sides = cut_by_month(windows, ["m/2000-01", "m/2000-02"], ["m/2001-01"])

    assert [w.window.position for w in sides.pool] == [0, 1]
    assert [w.window.position for w in sides.read] == [4, 5]
    assert sides.straddling == {2}


def test_months_on_both_sides_are_refused() -> None:
    with pytest.raises(SystemExit, match="both teach and read"):
        cut_by_month([], ["m/2000-01"], ["m/2000-01"])


def test_a_draw_weighs_every_event_alike_however_long_and_repeats_under_its_seed() -> None:
    events = {1: [0], 2: list(range(1, 100))}
    draw = BalancedDraw(events, [200, 201], seed=3)

    indices, targets = draw.batch(4000)

    assert targets == [1.0] * 2000 + [0.0] * 2000
    assert indices[:2000].count(0) == pytest.approx(1000, abs=120)
    assert set(indices[2000:]) == {200, 201}
    assert BalancedDraw(events, [200, 201], seed=3).batch(4000)[0] == indices
    assert BalancedDraw(events, [200, 201], seed=4).batch(4000)[0] != indices


def test_a_draw_with_nothing_to_learn_from_is_refused() -> None:
    with pytest.raises(SystemExit, match="needs a taught window"):
        BalancedDraw({}, [1], seed=1)


def separable() -> tuple[nn.Module, BalancedDraw, torch.Tensor]:
    torch.manual_seed(1)
    rows = torch.cat([torch.randn(40, 3) + 2.0, torch.randn(40, 3) - 2.0])
    draw = BalancedDraw({1: list(range(20)), 2: list(range(20, 40))}, list(range(40, 80)), seed=1)
    return nn.Linear(3, 1), draw, rows


def test_the_loop_learns_and_says_how_far_the_weights_moved() -> None:
    model, draw, rows = separable()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    early, late = fine_tuned(
        model, lambda at: model(rows[list(at)]).squeeze(-1), draw, SCHEDULE, device="cpu"
    )

    assert late < early
    assert moved_share(before, model) > 0.0


def test_a_loss_that_stops_being_finite_is_refused() -> None:
    model, draw, rows = separable()

    with pytest.raises(DivergedAdaptationError, match="the loss of step 0"):
        fine_tuned(
            model,
            lambda at: model(rows[list(at)]).squeeze(-1) * float("nan"),
            draw,
            SCHEDULE,
            device="cpu",
        )


def ranking(seed: int, lift: float) -> AnomalyRanking:
    draws = random.Random(seed)
    return AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(unit=UnitKey(f"u{unit:02d}"), position=0, ends_at=0.0),
            target=float(unit + 1) if unit < 5 else 0.0,
            predicted=draws.random() + (lift if unit < 5 else 0.0),
        )
        for unit in range(30)
    )


def test_a_large_gain_passes_and_two_pretrainings_that_disagree_do_not() -> None:
    initialisation = {seed: ranking(seed, 0.0) for seed in (1, 2, 3)}
    alike = {
        name: {seed: ranking(seed + at, 2.0) for seed in (1, 2, 3)}
        for at, name in enumerate(("base1", "base2"))
    }
    apart = {
        "base1": {seed: ranking(seed, 2.0) for seed in (1, 2, 3)},
        "base2": {seed: ranking(seed, -2.0) for seed in (1, 2, 3)},
    }

    gained = gain_row("ft@0.001", 5, alike, initialisation, bootstrap=QUICK)
    split = gain_row("ft@0.001", 5, apart, initialisation, bootstrap=QUICK)

    assert gained.passes
    assert gained.gain > gained.threshold
    assert not split.passes
    assert split.threshold > gained.threshold


def test_holm_keeps_the_family_to_fine_tuning_and_steps_down_from_the_smallest_p() -> None:
    row = gain_row(
        "ft@0.001",
        5,
        {"base1": {1: ranking(1, 2.0)}},
        {1: ranking(1, 0.0)},
        bootstrap=QUICK,
    )
    family = [
        replace(row, arm="ft@0.001", events=5, p_value=0.001),
        replace(row, arm="ft@0.001", events=10, p_value=0.03),
        replace(row, arm="ft@0.0001", events=5, p_value=0.03),
        replace(row, arm="direction", events=5, p_value=0.0001),
    ]

    judged = corrected(family)

    assert [each.rejected for each in judged] == [True, False, False, False]


def test_sides_holding_other_seeds_are_refused() -> None:
    with pytest.raises(SystemExit, match="same seeds"):
        gain_row(
            "ft",
            5,
            {"base1": {1: ranking(1, 1.0)}},
            {2: ranking(2, 0.0)},
            bootstrap=QUICK,
        )


def test_answers_read_back_as_written(tmp_path: Path) -> None:
    written = [
        WindowPrediction(
            window=TaskWindow(unit=UnitKey("m/2003-01"), position=7, ends_at=7.5),
            target=2.0,
            predicted=-0.25,
        )
    ]
    path = tmp_path / "answers.csv"

    write_answers(path, written)

    assert read_answers(path) == written
    assert path.read_text().splitlines()[0].split(",") == list(ANSWER_COLUMNS)


def test_the_tables_are_read_from_the_files_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "answers").mkdir()
    with (tmp_path / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        for name, lift in (("base1", 2.0), ("base2", 2.0), ("initialisation", 0.0)):
            for seed in (1, 2):
                cell = f"ft-{name}-k5-s{seed}"
                writer.writerow((cell, "ft@0.001", name, 5, seed, 9, 40, 1.0, 0.7, 0.2, 0.1))
                write_answers(
                    tmp_path / "answers" / f"{cell}.csv",
                    (
                        WindowPrediction(
                            window=TaskWindow(
                                unit=UnitKey(f"u{unit:02d}"), position=0, ends_at=0.0
                            ),
                            target=float(unit + 1) if unit < 5 else 0.0,
                            predicted=random.Random(seed * 100 + unit).random()
                            + (lift if unit < 5 else 0.0),
                        )
                        for unit in range(30)
                    ),
                )

    main(["read", "--dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "base1" in out
    assert "yes" in out or "no" in out
    assert (tmp_path / "gains.md").exists()
    assert "| ft@0.001 | 5 | initialisation |" in (tmp_path / "areas.md").read_text()
    assert "ft-base1-k5-s1" in (tmp_path / "mechanics.md").read_text()


def test_a_directory_without_a_cell_is_refused(tmp_path: Path) -> None:
    (tmp_path / "cells.csv").write_text(",".join(CELL_COLUMNS) + "\n")

    with pytest.raises(SystemExit, match="no cell"):
        main(["read", "--dir", str(tmp_path)])
