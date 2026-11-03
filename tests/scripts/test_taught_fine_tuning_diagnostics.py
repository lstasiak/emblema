"""The diagnostics' cuts and tables on small data; no backbone and no registry."""

import csv
import json
import random
from pathlib import Path

import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.taught_class_report import OTHER_HALF, TAUGHT_HALF
from scripts.taught_fine_tuning_diagnostics import (
    class_rows,
    class_table,
    control_sides,
    main,
    read_tables,
    refuse_other_conditions,
    split_by_class,
)
from scripts.taught_fine_tuning_pilot import CELL_COLUMNS, write_answers


def labelled(unit: str, position: int, target: float) -> LabelledWindow:
    return LabelledWindow(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
        target=target,
    )


def predicted(unit: str, position: int, target: float, score: float) -> WindowPrediction:
    return WindowPrediction(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
        target=target,
        predicted=score,
    )


def test_a_control_teaches_a_long_events_first_half_and_reads_its_second() -> None:
    pool = [
        labelled("m/01", 0, 0.0),
        labelled("m/02", 1, 0.0),
        labelled("m/03", 2, 0.0),
        labelled("m/04", 3, 0.0),
        labelled("m/02", 4, 1.0),
        labelled("m/02", 5, 1.0),
        labelled("m/04", 6, 1.0),
        labelled("m/04", 7, 1.0),
        labelled("m/03", 8, 2.0),
    ]

    sides = control_sides(pool, least_windows=4)

    assert sides.events == 1
    assert [w.window.position for w in sides.taught] == [4, 5]
    assert sorted(w.window.position for w in sides.teach) == [0, 2, 4, 5]
    assert sorted(w.window.position for w in sides.read) == [1, 3, 6, 7]


def test_a_month_holding_a_read_window_is_read_whole_so_no_event_is_read_against_itself() -> None:
    pool = [
        labelled("m/01", 0, 0.0),
        labelled("m/01", 1, 1.0),
        labelled("m/01", 2, 1.0),
        labelled("m/02", 3, 0.0),
        labelled("m/03", 4, 0.0),
    ]

    sides = control_sides(pool, least_windows=2)

    read_months = {str(w.window.unit) for w in sides.read if w.target}
    normal_months = {str(w.window.unit) for w in sides.read if not w.target}
    assert read_months <= normal_months
    assert not {str(w.window.unit) for w in sides.teach if not w.target} & normal_months
    assert {w.window.position for w in sides.teach if not w.target} == {4}


def test_no_month_left_to_teach_normal_windows_is_refused() -> None:
    pool = [labelled("m/01", 0, 0.0), labelled("m/01", 1, 1.0), labelled("m/01", 2, 1.0)]

    with pytest.raises(SystemExit, match="no normal window teaches"):
        control_sides(pool, least_windows=2)


def test_a_control_without_a_long_event_is_refused() -> None:
    with pytest.raises(SystemExit, match="no event holds 3 windows"):
        control_sides([labelled("m/01", 0, 1.0), labelled("m/01", 1, 0.0)], least_windows=3)


def test_each_half_holds_the_normal_windows_and_the_events_of_its_classes() -> None:
    answers = [
        predicted("u", 0, 0.0, 0.1),
        predicted("u", 1, 1.0, 0.2),
        predicted("u", 2, 2.0, 0.3),
    ]

    split = split_by_class(answers, {1: "a", 2: "b"}, frozenset({"a"}))

    assert [each.window.position for each in split[TAUGHT_HALF]] == [0, 1]
    assert [each.window.position for each in split[OTHER_HALF]] == [0, 2]


def untaught_answers(directory: Path, name: str, scores: tuple[float, float]) -> None:
    (directory / "answers").mkdir()
    write_answers(
        directory / "answers" / f"untaught-{name}-k0-s1.csv",
        [predicted("u", 0, 0.0, scores[0]), predicted("u", 1, 1.0, scores[1])],
    )


def test_a_half_is_read_as_a_gain_over_the_same_encoder_untaught(tmp_path: Path) -> None:
    pilot, untaught = tmp_path / "pilot", tmp_path / "untaught"
    pilot.mkdir()
    untaught.mkdir()
    (pilot / "answers").mkdir()
    write_answers(
        pilot / "answers" / "c.csv",
        [predicted("u", 0, 0.0, 0.1), predicted("u", 1, 1.0, 0.9)],
    )
    untaught_answers(untaught, "base1", (0.9, 0.1))
    cell = {"cell": "c", "arm": "ft@0.001", "backbone": "base1", "events": "5", "seed": "1"}

    rows = class_rows([cell], pilot, untaught, {1: "a"}, {(5, 1): frozenset({"a"})})

    assert [row[4] for row in rows] == [TAUGHT_HALF]
    assert rows[0][5] == 1
    assert rows[0][6] == pytest.approx(1.0)
    assert rows[0][7] == pytest.approx(0.5)
    assert rows[0][8] == pytest.approx(0.5)


def test_an_untaught_cell_over_other_windows_is_refused(tmp_path: Path) -> None:
    pilot, untaught = tmp_path / "pilot", tmp_path / "untaught"
    pilot.mkdir()
    untaught.mkdir()
    (pilot / "answers").mkdir()
    write_answers(pilot / "answers" / "c.csv", [predicted("elsewhere", 9, 1.0, 0.9)])
    untaught_answers(untaught, "base1", (0.9, 0.1))
    cell = {"cell": "c", "arm": "ft@0.001", "backbone": "base1", "events": "5", "seed": "1"}

    with pytest.raises(SystemExit, match="scored other windows"):
        class_rows([cell], pilot, untaught, {1: "a"}, {(5, 1): frozenset({"a"})})


def test_the_class_table_sets_the_two_gains_side_by_side() -> None:
    rows = [
        ("ft@0.001", "base1", 5, 1, TAUGHT_HALF, 5, 0.4, 0.1, 0.3, 0.5),
        ("ft@0.001", "base1", 5, 1, OTHER_HALF, 57, 0.2, 0.1, 0.1, 0.4),
    ]

    assert "| ft@0.001 | 5 | base1 | +0.300 | +0.100 | +0.200 |" in class_table(rows)


def test_a_run_under_other_conditions_takes_no_more_cells(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps({"least_windows": 20, "rate": 0.001}))

    refuse_other_conditions(path, {"least_windows": 20, "rate": 0.001})
    with pytest.raises(SystemExit, match="least_windows"):
        refuse_other_conditions(path, {"least_windows": 10, "rate": 0.001})
    refuse_other_conditions(tmp_path / "absent.json", {"least_windows": 10})


def test_the_curves_full_length_stands_beside_it_from_the_pilots_directory(tmp_path: Path) -> None:
    here, pilot = tmp_path / "here", tmp_path / "pilot"
    for directory, arm, steps in ((here, "ft@0.001-steps200", 200), (pilot, "ft@0.001", 2000)):
        directory.mkdir()
        (directory / "answers").mkdir()
        cell = f"{arm}-base1-k5-s1"
        write_answers(
            directory / "answers" / f"{cell}.csv",
            [predicted("u", 0, 0.0, 0.1), predicted("u", 1, 1.0, 0.9)],
        )
        with (directory / "cells.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CELL_COLUMNS)
            writer.writerow((cell, arm, "base1", 5, 1, 3, steps, 1.0, 0.7, 0.2, 0.1))

    text = read_tables(here, beside=pilot)

    assert "| ft@0.001-steps200 | 5 | base1 |" in text
    assert "| ft@0.001-steps2000 | 5 | base1 |" in text


def write_cell(
    directory: Path, rows: list[tuple[object, ...]], name: str, seed: int, lift: float
) -> None:
    cell = f"control-{name}-k5-s{seed}"
    rows.append((cell, "control", name, 5, seed, 3, 40, 1.0, 0.7, 0.2, 0.1))
    draws = random.Random(seed * 100)
    write_answers(
        directory / "answers" / f"{cell}.csv",
        [
            predicted(
                f"u{unit:02d}",
                0,
                float(unit + 1) if unit < 5 else 0.0,
                draws.random() + (lift if unit < 5 else 0.0),
            )
            for unit in range(30)
        ],
    )


def test_the_tables_describe_every_arm_and_pass_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "answers").mkdir()
    rows: list[tuple[object, ...]] = []
    for name, lift in (("base1", 2.0), ("base2", 2.0), ("initialisation", 0.0)):
        for seed in (1, 2):
            write_cell(tmp_path, rows, name, seed, lift)
    with (tmp_path / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        writer.writerows(rows)

    main(["read", "--dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "| control | 5 | base1 |" in out
    assert "Pretrained less initialisation" in out
    assert "pass" not in out.lower()
    assert (tmp_path / "tables.md").exists()


def test_a_directory_without_a_cell_is_refused(tmp_path: Path) -> None:
    (tmp_path / "cells.csv").write_text(",".join(CELL_COLUMNS) + "\n")

    with pytest.raises(SystemExit, match="no cell"):
        main(["read", "--dir", str(tmp_path)])
