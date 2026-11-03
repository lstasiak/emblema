"""The pilot's cut, draw, loop and reading on small data; no backbone and no registry."""

import csv
import json
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
    Bar,
    adopt,
    corrected,
    fine_tuned,
    gain_row,
    main,
    missing_cells,
    moved_share,
    read_answers,
    refuse_another_commit,
    spread_of,
    taught_spans,
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


def test_a_draw_weighs_every_event_alike_however_long_and_repeats_under_its_seed() -> None:
    events = {1: [0], 2: list(range(1, 100))}
    draw = BalancedDraw(events, [200, 201], seed=3)

    indices, targets = draw.batch(4000)

    assert targets == [1.0] * 2000 + [0.0] * 2000
    assert indices[:2000].count(0) == pytest.approx(1000, abs=120)
    assert set(indices[2000:]) == {200, 201}
    assert BalancedDraw(events, [200, 201], seed=3).batch(4000)[0] == indices
    assert BalancedDraw(events, [200, 201], seed=4).batch(4000)[0] != indices


def test_taught_windows_are_found_in_the_pool_by_event_and_the_rest_are_the_others() -> None:
    def window(position: int, target: float) -> LabelledWindow:
        return LabelledWindow(
            window=TaskWindow(unit=UnitKey("u"), position=position, ends_at=float(position)),
            target=target,
        )

    pool = [window(10, 0.0), window(11, 1.0), window(12, 1.0), window(13, 2.0), window(14, 0.0)]

    spans, others = taught_spans(pool, [window(12, 1.0), window(11, 1.0)])

    assert spans == {1: [2, 1]}
    assert others == [0, 3, 4]


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

    gained = gain_row(
        "ft@0.001",
        5,
        alike,
        initialisation,
        spread=spread_of([alike], bootstrap=QUICK),
        bootstrap=QUICK,
    )
    split = gain_row(
        "ft@0.001",
        5,
        apart,
        initialisation,
        spread=spread_of([apart], bootstrap=QUICK),
        bootstrap=QUICK,
    )

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
        spread=0.014,
        bootstrap=QUICK,
    )
    family = [
        replace(row, arm="ft@0.001", events=5, p_value=0.001),
        replace(row, arm="ft@0.001", events=10, p_value=0.03),
        replace(row, arm="ft@0.0001", events=5, p_value=0.03),
        replace(row, arm="direction", events=5, p_value=0.0001),
    ]

    judged = corrected(family, family_size=3)

    assert [each.rejected for each in judged] == [True, False, False, False]
    assert [each.rejected for each in corrected(family[:1], family_size=100)] == [False]


def test_readings_that_agree_are_held_to_the_spread_of_those_that_do_not() -> None:
    agree = {name: {seed: ranking(seed, 2.0) for seed in (1, 2, 3)} for name in ("a", "b")}
    disagree = {
        "a": {seed: ranking(seed, 2.0) for seed in (1, 2, 3)},
        "b": {seed: ranking(seed, -2.0) for seed in (1, 2, 3)},
    }

    alone = spread_of([agree], bootstrap=QUICK)
    pooled = spread_of([agree, disagree], bootstrap=QUICK)

    assert alone == 0.014
    assert pooled > alone


def test_a_reading_below_minus_t_with_its_interval_below_zero_is_harm() -> None:
    row = gain_row(
        "ft@0.001",
        5,
        {"base1": {1: ranking(1, 2.0)}},
        {1: ranking(1, 0.0)},
        spread=0.014,
        bootstrap=QUICK,
    )

    assert replace(row, gain=-0.2, high=-0.1, threshold=0.05).outcome == "harm"
    assert replace(row, gain=0.2, low=0.1, threshold=0.05, rejected=True).outcome == "large gain"
    # Near nought with a wide bar: the least gain is not ruled out, so nothing is closed.
    wide = replace(row, gain=0.0, low=-0.05, high=0.04, threshold=0.08, rejected=False)
    assert wide.outcome == "inconclusive"
    # Near nought with a narrow bar and months that agree: the least gain is ruled out.
    narrow = replace(wide, threshold=0.03)
    assert narrow.outcome == "closed"
    # The months' interval holds it open even under a narrow bar.
    assert replace(narrow, high=0.06).outcome == "inconclusive"


def test_sides_holding_other_seeds_are_refused() -> None:
    with pytest.raises(SystemExit, match="same seeds"):
        gain_row(
            "ft",
            5,
            {"base1": {1: ranking(1, 1.0)}},
            {2: ranking(2, 0.0)},
            spread=0.014,
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
    grid(tmp_path, seeds=[1, 2])
    with (tmp_path / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        for name, lift in (("base1", 2.0), ("base2", 2.0), ("initialisation", 0.0)):
            for seed in (1, 2):
                cell = f"ft@0.001-{name}-k5-s{seed}"
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
    assert "large gain" in out or "inconclusive" in out or "closed" in out
    assert (tmp_path / "gains.md").exists()
    assert "| ft@0.001 | 5 | initialisation |" in (tmp_path / "areas.md").read_text()
    assert "ft@0.001-base1-k5-s1" in (tmp_path / "mechanics.md").read_text()


def grid(directory: Path, *, seeds: list[int]) -> None:
    (directory / "conditions.json").write_text(
        json.dumps(
            {
                "commit": "abc",
                "encoders": ["base1", "base2"],
                "rates": [0.001],
                "events": [5],
                "seeds": seeds,
            }
        )
    )


def test_a_grid_not_whole_is_read_only_when_asked(tmp_path: Path) -> None:
    grid(tmp_path, seeds=[1, 2, 3])

    assert missing_cells(
        json.loads((tmp_path / "conditions.json").read_text()),
        [f"ft@0.001-{name}-k5-s{seed}" for name in ("base1", "base2") for seed in (1, 2, 3)]
        + ["ft@0.001-initialisation-k5-s1", "ft@0.001-initialisation-k5-s2"],
    ) == ["ft@0.001-initialisation-k5-s3"]


def test_a_run_begun_at_another_commit_takes_no_more_cells(tmp_path: Path) -> None:
    grid(tmp_path, seeds=[1])
    path = tmp_path / "conditions.json"

    refuse_another_commit(path, "abc")
    with pytest.raises(SystemExit, match="began at abc"):
        refuse_another_commit(path, "def")
    refuse_another_commit(tmp_path / "absent.json", "def")


def test_a_directory_without_a_cell_is_refused(tmp_path: Path) -> None:
    grid(tmp_path, seeds=[1])
    (tmp_path / "cells.csv").write_text(",".join(CELL_COLUMNS) + "\n")

    with pytest.raises(SystemExit, match="no cell"):
        main(["read", "--dir", str(tmp_path)])


EXPORT_COLUMNS = (
    "campaign",
    "candidate",
    "budget",
    "seed",
    "unit",
    "position",
    "ends_at",
    "target",
    "predicted",
)


def export(path: Path, *rows: tuple[str, str, int, float, float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPORT_COLUMNS)
        for campaign, candidate, seed, target, predicted in rows:
            writer.writerow((campaign, candidate, "all", seed, "u", 0, 0.0, target, predicted))
            writer.writerow((campaign, candidate, "all", seed, "v", 1, 1.0, 0.0, -predicted))


BAR = Bar(arm="direction", encoder="base1", campaign="c1", candidate="knn@a=1", events=5)


def test_a_bar_is_copied_in_at_its_seeds_and_only_once(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    predictions = tmp_path / "export.csv"
    export(
        predictions,
        ("c1", "knn@a=1", 1, 1.0, 0.5),
        ("c1", "knn@a=1", 2, 1.0, 0.6),
        ("c2", "x", 1, 1.0, 0.1),
    )

    assert adopt(predictions, tmp_path, [BAR], [1]) == 1
    assert adopt(predictions, tmp_path, [BAR], [1]) == 0

    copied = read_answers(tmp_path / "answers" / "direction-base1-k5-s1.csv")
    assert [each.predicted for each in copied] == [0.5, -0.5]
    assert not (tmp_path / "answers" / "direction-base1-k5-s2.csv").exists()


def test_a_bar_without_the_seed_is_refused(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    predictions = tmp_path / "export.csv"
    export(predictions, ("c1", "knn@a=1", 1, 1.0, 0.5))

    with pytest.raises(SystemExit, match="holds no seed 2"):
        adopt(predictions, tmp_path, [BAR], [2])


def test_a_bar_that_scored_other_windows_pairs_nothing_and_is_refused(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    write_answers(
        tmp_path / "answers" / "ft@0.001-base1-k5-s1.csv",
        [
            WindowPrediction(
                window=TaskWindow(unit=UnitKey("elsewhere"), position=9, ends_at=9.0),
                target=0.0,
                predicted=0.1,
            )
        ],
    )
    predictions = tmp_path / "export.csv"
    export(predictions, ("c1", "knn@a=1", 1, 1.0, 0.5))

    with pytest.raises(SystemExit, match="scored other windows"):
        adopt(predictions, tmp_path, [BAR], [1])
