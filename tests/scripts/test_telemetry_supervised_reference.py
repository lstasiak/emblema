"""The reference's cut by month, its grid and lanes, its choice, its bars and its reading."""

import csv
import json
import random
from pathlib import Path
from statistics import stdev

import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.taught_fine_tuning_pilot import CELL_COLUMNS, write_answers
from scripts.telemetry_supervised_reference import (
    SEEN_HALF,
    SELECTION_SUFFIX,
    UNSEEN_HALF,
    Plan,
    adopt,
    arm_name,
    cells_of,
    choose,
    choose_from,
    class_rows,
    class_table,
    lane_of,
    lane_table,
    main,
    reference_row,
    refit_plans,
    refuse_other_conditions,
    select_plans,
    split_arm,
    split_by_seen,
    time_sides,
)

QUICK = PairedUnitBootstrap(resamples=50, seed=1)


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


def test_the_earlier_months_teach_and_the_later_ones_are_read() -> None:
    pool = [
        labelled("m/2003-12", 0, 0.0),
        labelled("m/2003-12", 1, 1.0),
        labelled("m/2004-01", 2, 0.0),
        labelled("m/2004-01", 3, 2.0),
        labelled("m/2004-02", 4, 3.0),
    ]

    sides = time_sides(pool, select_from="2004-01")

    assert [w.window.position for w in sides.teach] == [0, 1]
    assert [w.window.position for w in sides.taught] == [1]
    assert [w.window.position for w in sides.read] == [2, 3, 4]
    assert (sides.events_taught, sides.events_read, sides.events_dropped) == (1, 2, 0)


def test_an_event_on_both_sides_teaches_its_earlier_windows_and_is_not_read() -> None:
    pool = [
        labelled("m/2003-12", 0, 0.0),
        labelled("m/2003-12", 1, 1.0),
        labelled("m/2004-01", 2, 1.0),
        labelled("m/2004-01", 3, 0.0),
        labelled("m/2004-02", 4, 2.0),
    ]

    sides = time_sides(pool, select_from="2004-01")

    assert [w.window.position for w in sides.taught] == [1]
    assert [w.window.position for w in sides.read] == [3, 4]
    assert (sides.events_taught, sides.events_read, sides.events_dropped) == (1, 1, 1)


@pytest.mark.parametrize("select_from", ["2003-01", "2005-01"])
def test_a_cut_leaving_a_side_without_an_event_is_refused(select_from: str) -> None:
    pool = [labelled("m/2003-12", 0, 1.0), labelled("m/2004-01", 1, 2.0)]

    with pytest.raises(SystemExit, match="without an event"):
        time_sides(pool, select_from=select_from)


def test_an_arms_name_states_its_section_rate_and_length() -> None:
    assert split_arm(arm_name("select", 0.0003, 16000)) == ("select", 0.0003, 16000)
    assert split_arm("refit-ft@0.001-steps500") == ("refit", 0.001, 500)
    with pytest.raises(ValueError, match="names no fine-tuning"):
        split_arm("direction")
    with pytest.raises(ValueError, match="names no fine-tuning"):
        split_arm("ft@0.001")


def test_the_grid_runs_the_longest_first_and_lanes_take_every_nth_cell() -> None:
    plans = select_plans(
        [0.001, 0.0003], [500, 2000], ["base1", "initialisation"], [1, 2], events=7
    )

    assert len(plans) == 16
    assert [plan.steps for plan in plans[:8]] == [2000] * 8
    assert plans[0].cell == "select-ft@0.001-steps2000-base1-k7-s1"
    assert [plan.cell for plan in lane_of(plans, "2/3")] == [plan.cell for plan in plans[1::3]]
    assert sorted(
        plan.cell for lane in ("1/3", "2/3", "3/3") for plan in lane_of(plans, lane)
    ) == sorted(plan.cell for plan in plans)
    assert lane_of(plans, None) == plans


@pytest.mark.parametrize("lane", ["0/3", "4/3", "a/b", "2"])
def test_a_lane_that_is_not_i_of_n_is_refused(lane: str) -> None:
    with pytest.raises(SystemExit, match="lane is i/n"):
        lane_of([], lane)


def areas_of(**runs: tuple[float, ...]) -> dict[tuple[str, float, int], dict[int, float]]:
    """Runs named ``r<rate>_s<steps>`` with their areas per seed, all under one encoder."""
    areas = {}
    for name, seeds in runs.items():
        rate, steps = name.removeprefix("r").split("_s")
        areas[("initialisation", float(rate.replace("p", ".")), int(steps))] = dict(
            enumerate(seeds, start=1)
        )
    return areas


def test_the_best_mean_chooses_and_the_longest_asks_the_grid_to_grow() -> None:
    choices = choose(
        areas_of(r0p001_s500=(0.1, 0.1), r0p001_s2000=(0.3, 0.3)), lengths=[500, 2000], bars={}
    )

    assert len(choices) == 1
    choice = choices[0]
    assert (choice.encoder, choice.rate, choice.steps) == ("initialisation", 0.001, 2000)
    assert choice.mean == pytest.approx(0.3)
    assert choice.per_seed == {1: 0.3, 2: 0.3}
    assert (choice.extend, choice.bar, choice.untrainable) == (True, None, None)


def test_a_shorter_run_within_a_hundredth_of_the_best_is_preferred() -> None:
    choices = choose(
        areas_of(r0p001_s500=(0.295, 0.295), r0p001_s2000=(0.3, 0.3), r0p001_s8000=(0.2, 0.2)),
        lengths=[500, 2000, 8000],
        bars={},
    )

    assert (choices[0].steps, choices[0].extend) == (500, False)


def test_among_equal_lengths_the_higher_mean_wins_and_then_the_rate_in_force() -> None:
    choices = choose(
        areas_of(r0p001_s500=(0.30, 0.30), r0p0003_s500=(0.305, 0.305)), lengths=[500], bars={}
    )
    assert choices[0].rate == 0.0003

    choices = choose(
        areas_of(r0p001_s500=(0.30, 0.30), r0p0003_s500=(0.30, 0.30)), lengths=[500], bars={}
    )
    assert choices[0].rate == 0.001


def test_a_choice_below_the_direction_on_the_same_months_is_untrainable() -> None:
    areas = areas_of(r0p001_s500=(0.1, 0.1))

    trained = choose(areas, lengths=[500], bars={"initialisation": 0.05})
    untrained = choose(areas, lengths=[500], bars={"initialisation": 0.2})

    assert trained[0].untrainable is False
    assert untrained[0].untrainable is True


def test_runs_of_one_encoder_with_different_seeds_are_refused() -> None:
    areas = areas_of(r0p001_s500=(0.1, 0.1), r0p001_s2000=(0.1,))

    with pytest.raises(SystemExit, match="different seeds"):
        choose(areas, lengths=[500, 2000], bars={})


def write_conditions(directory: Path, **overrides: object) -> None:
    conditions: dict[str, object] = {
        "commit": "c",
        "encoders": ["base1", "initialisation"],
        "seeds": [1, 2],
        "rates": [0.001],
        "lengths": [500, 2000],
        "select_from": "2004-01",
        "events_in_pool": 9,
        "events_taught_in_selection": 7,
    }
    conditions.update(overrides)
    (directory / "conditions.json").write_text(json.dumps(conditions))


def month_answers(lift: float, *, seed: int, months: int = 6) -> list[WindowPrediction]:
    """Six months of scores, the first window of each an event, raised by ``lift``."""
    draws = random.Random(seed)
    return [
        predicted(
            f"m/2007-{month:02d}",
            month * 10 + at,
            float(month) if at == 0 else 0.0,
            draws.random() + (lift if at == 0 else 0.0),
        )
        for month in range(1, months + 1)
        for at in range(10)
    ]


def write_cell(
    directory: Path,
    rows: list[tuple[object, ...]],
    *,
    arm: str,
    encoder: str,
    events: int,
    seed: int,
    steps: int,
    lift: float,
    selection_lift: float | None = None,
) -> str:
    cell = f"{arm}-{encoder}-k{events}-s{seed}"
    rows.append((cell, arm, encoder, events, seed, 30, steps, 1.0, 0.7, 0.2, 0.1))
    write_answers(directory / "answers" / f"{cell}.csv", month_answers(lift, seed=seed))
    if selection_lift is not None:
        write_answers(
            directory / "answers" / f"{cell}{SELECTION_SUFFIX}.csv",
            month_answers(selection_lift, seed=seed + 50, months=3),
        )
    return cell


def write_rows(directory: Path, rows: list[tuple[object, ...]], name: str = "cells.csv") -> None:
    with (directory / name).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        writer.writerows(rows)


def test_the_choice_reads_the_selection_months_alone_and_writes_what_it_chose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "answers").mkdir()
    write_conditions(tmp_path)
    rows: list[tuple[object, ...]] = []
    for encoder in ("base1", "initialisation"):
        for steps, selection_lift, lift in ((500, 3.0, 0.0), (2000, 0.0, 3.0)):
            for seed in (1, 2):
                write_cell(
                    tmp_path,
                    rows,
                    arm=arm_name("select", 0.001, steps),
                    encoder=encoder,
                    events=7,
                    seed=seed,
                    steps=steps,
                    lift=lift,
                    selection_lift=selection_lift,
                )
    write_rows(tmp_path, rows[: len(rows) // 2], "cells-1-of-2.csv")
    write_rows(tmp_path, rows[len(rows) // 2 :], "cells-2-of-2.csv")

    with pytest.raises(SystemExit, match="4 bars are missing"):
        main(["choose", "--dir", str(tmp_path)])
    main(["choose", "--dir", str(tmp_path), "--partial"])

    chosen = json.loads((tmp_path / "chosen.json").read_text())
    assert [(choice["encoder"], choice["steps"], choice["extend"]) for choice in chosen] == [
        ("base1", 500, False),
        ("initialisation", 500, False),
    ]
    assert "| base1 | 0.001 | 500 |" in capsys.readouterr().out
    assert (tmp_path / "chosen.md").exists()


def test_the_bars_on_the_selection_months_enter_the_choice(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    (tmp_path / "bars" / "answers").mkdir(parents=True)
    write_conditions(tmp_path, encoders=["initialisation"])
    rows: list[tuple[object, ...]] = []
    write_cell(
        tmp_path,
        rows,
        arm=arm_name("select", 0.001, 500),
        encoder="initialisation",
        events=7,
        seed=1,
        steps=500,
        lift=0.0,
        selection_lift=0.0,
    )
    write_rows(tmp_path, rows)
    bars: list[tuple[object, ...]] = []
    cell = "select-direction-initialisation-k7-s1"
    bars.append((cell, "select-direction", "initialisation", 7, 1, 30, 0, 1.0, "", "", ""))
    write_answers(
        tmp_path / "bars" / "answers" / f"{cell}.csv", month_answers(3.0, seed=9, months=3)
    )
    write_rows(tmp_path / "bars", bars)

    with pytest.raises(SystemExit, match="selection cells are missing"):
        choose_from(tmp_path)
    choices = choose_from(tmp_path, partial=True)

    assert choices[0].bar == pytest.approx(1.0)
    assert choices[0].untrainable is True
    assert "untrainable" in (tmp_path / "chosen.md").read_text()


def test_a_whole_grid_with_its_bars_is_chosen_without_asking(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    (tmp_path / "bars" / "answers").mkdir(parents=True)
    write_conditions(tmp_path, encoders=["initialisation"], seeds=[1], lengths=[500])
    rows: list[tuple[object, ...]] = []
    write_cell(
        tmp_path,
        rows,
        arm=arm_name("select", 0.001, 500),
        encoder="initialisation",
        events=7,
        seed=1,
        steps=500,
        lift=0.0,
        selection_lift=0.5,
    )
    write_rows(tmp_path, rows)
    cell = "select-direction-initialisation-k7-s1"
    write_answers(
        tmp_path / "bars" / "answers" / f"{cell}.csv", month_answers(0.0, seed=9, months=3)
    )
    write_rows(
        tmp_path / "bars",
        [(cell, "select-direction", "initialisation", 7, 1, 30, 0, 1.0, "", "", "")],
    )

    choices = choose_from(tmp_path)

    assert (choices[0].steps, choices[0].untrainable) == (500, False)


def test_a_choice_without_a_selection_cell_is_refused(tmp_path: Path) -> None:
    write_conditions(tmp_path)
    write_rows(tmp_path, [])

    with pytest.raises(SystemExit, match="no selection cell"):
        choose_from(tmp_path)


def test_the_refit_follows_the_choice_and_waits_for_a_grid_asked_to_grow(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="choose first"):
        refit_plans(tmp_path, [1], events=9)

    (tmp_path / "chosen.json").write_text(
        json.dumps(
            [
                {"encoder": "base1", "rate": 0.001, "steps": 500, "extend": False},
                {"encoder": "initialisation", "rate": 0.0003, "steps": 2000, "extend": False},
            ]
        )
    )
    plans = refit_plans(tmp_path, [1, 2], events=9)
    assert [plan.cell for plan in plans] == [
        "refit-ft@0.001-steps500-base1-k9-s1",
        "refit-ft@0.001-steps500-base1-k9-s2",
        "refit-ft@0.0003-steps2000-initialisation-k9-s1",
        "refit-ft@0.0003-steps2000-initialisation-k9-s2",
    ]

    (tmp_path / "chosen.json").write_text(
        json.dumps([{"encoder": "base1", "rate": 0.001, "steps": 2000, "extend": True}])
    )
    with pytest.raises(SystemExit, match="grow once"):
        refit_plans(tmp_path, [1], events=9)


def test_a_run_keeps_its_conditions_and_lets_its_lengths_grow(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    conditions: dict[str, object] = {"commit": "c", "lengths": [500, 2000], "seeds": [1]}
    refuse_other_conditions(path, conditions)
    path.write_text(json.dumps(conditions))

    refuse_other_conditions(path, {**conditions, "lengths": [500, 2000, 4000]})
    with pytest.raises(SystemExit, match="grow, not shrink"):
        refuse_other_conditions(path, {**conditions, "lengths": [500]})
    with pytest.raises(SystemExit, match=r"other conditions: \['seeds'\]"):
        refuse_other_conditions(path, {**conditions, "seeds": [1, 2]})


def test_a_lane_records_into_its_own_table(tmp_path: Path) -> None:
    assert lane_table(tmp_path, None) == tmp_path / "cells.csv"
    assert lane_table(tmp_path, "2/3") == tmp_path / "cells-2-of-3.csv"
    assert (tmp_path / "cells-2-of-3.csv").read_text().startswith("cell,arm,backbone")


def source_directory(tmp_path: Path, *, other_windows: bool = False) -> Path:
    source = tmp_path / "source"
    (source / "answers").mkdir(parents=True)
    rows: list[tuple[object, ...]] = []
    for encoder in ("base1", "initialisation"):
        for seed in (1, 2):
            cell = f"ceiling-direction-{encoder}-k9-s{seed}"
            rows.append((cell, "ceiling-direction", encoder, 9, seed, 30, 0, 1.0, "", "", ""))
            answers = month_answers(0.0, seed=seed)
            if other_windows:
                answers = answers[1:]
            write_answers(source / "answers" / f"{cell}.csv", answers)
    cell = "trees-summaries-k9-s1"
    rows.append((cell, "trees", "summaries", 9, 1, 30, 0, 1.0, "", "", ""))
    write_answers(source / "answers" / f"{cell}.csv", month_answers(0.5, seed=7))
    write_rows(source, rows)
    return source


def test_the_bars_are_adopted_renamed_and_only_once(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    source = source_directory(tmp_path)

    count = adopt(tmp_path, source, {"ceiling-direction": "direction", "trees": "trees"})

    assert count == 5
    assert (tmp_path / "answers" / "direction-base1-k9-s1.csv").exists()
    assert adopt(tmp_path, source, {"ceiling-direction": "direction"}) == 0
    with pytest.raises(SystemExit, match="holds no cell of combined"):
        adopt(tmp_path, source, {"combined": "combined"})


def test_a_bar_over_other_windows_than_the_cells_here_is_refused(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    rows: list[tuple[object, ...]] = []
    write_cell(
        tmp_path,
        rows,
        arm=arm_name("refit", 0.001, 500),
        encoder="initialisation",
        events=9,
        seed=1,
        steps=500,
        lift=1.0,
    )
    write_rows(tmp_path, rows)
    source = source_directory(tmp_path, other_windows=True)

    with pytest.raises(SystemExit, match="scored other windows"):
        adopt(tmp_path, source, {"ceiling-direction": "direction"})


def rankings(lift: float, seeds: tuple[int, ...]) -> dict[int, AnomalyRanking]:
    return {seed: AnomalyRanking.of(month_answers(lift, seed=seed)) for seed in seeds}


def test_a_refit_well_above_the_direction_reads_clearly_above() -> None:
    row = reference_row(
        "initialisation", rankings(4.0, (1, 2, 3)), rankings(0.0, (1, 2, 3)), bootstrap=QUICK
    )

    assert row.difference > 0.3
    assert row.low > 0.0
    assert row.outcome == "clearly above"
    differences = [
        row.areas[seed] - rankings(0.0, (seed,))[seed].average_precision for seed in (1, 2, 3)
    ]
    assert row.threshold == pytest.approx(2.0 * stdev(differences) / 3**0.5)


def test_a_refit_well_below_the_direction_reads_below_and_alike_reads_around() -> None:
    below = reference_row(
        "initialisation", rankings(0.0, (1, 2, 3)), rankings(4.0, (1, 2, 3)), bootstrap=QUICK
    )
    around = reference_row(
        "initialisation", rankings(2.0, (1, 2, 3)), rankings(2.0, (1, 2, 3)), bootstrap=QUICK
    )

    assert below.outcome == "below"
    assert around.outcome == "around"
    assert around.difference == pytest.approx(0.0)


def test_a_direction_lacking_a_seed_the_refit_holds_is_refused() -> None:
    with pytest.raises(SystemExit, match="lacks a seed"):
        reference_row("base1", rankings(1.0, (1, 2)), rankings(1.0, (1, 3)), bootstrap=QUICK)
    reference_row("base1", rankings(1.0, (1,)), rankings(1.0, (1, 3)), bootstrap=QUICK)


def reading_directory(tmp_path: Path, *, whole: bool = True) -> Path:
    (tmp_path / "answers").mkdir()
    write_conditions(tmp_path)
    rows: list[tuple[object, ...]] = []
    for encoder in ("base1", "initialisation"):
        for seed in (1, 2):
            write_cell(
                tmp_path,
                rows,
                arm=arm_name("select", 0.001, 500),
                encoder=encoder,
                events=7,
                seed=seed,
                steps=500,
                lift=0.5,
                selection_lift=0.5,
            )
            if whole or seed == 1:
                write_cell(
                    tmp_path,
                    rows,
                    arm=arm_name("refit", 0.001, 500),
                    encoder=encoder,
                    events=9,
                    seed=seed,
                    steps=500,
                    lift=2.0 if encoder == "initialisation" else 0.0,
                )
    write_rows(tmp_path, rows)
    (tmp_path / "chosen.json").write_text(
        json.dumps(
            [
                {"encoder": encoder, "rate": 0.001, "steps": 500, "extend": False}
                for encoder in ("base1", "initialisation")
            ]
        )
    )
    adopt(
        tmp_path, source_directory(tmp_path), {"ceiling-direction": "direction", "trees": "trees"}
    )
    return tmp_path


def test_the_reading_sets_the_refit_against_the_direction_and_describes_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = reading_directory(tmp_path)

    main(["read", "--dir", str(directory)])

    out = capsys.readouterr().out
    assert "| initialisation |" in out
    assert "| clearly above |" in out
    assert "| trees | summaries |" in out
    assert "Pretrained less initialisation" in out
    assert "| 0.001 | 500 | base1 |" in out
    assert (directory / "reference.md").exists()
    assert "select-ft@0.001-steps500-base1-k7-s1" in (directory / "mechanics.md").read_text()


def test_the_validation_months_stay_shut_until_every_refit_cell_is_in(tmp_path: Path) -> None:
    directory = reading_directory(tmp_path, whole=False)

    with pytest.raises(SystemExit, match="2 refit cells are missing"):
        main(["read", "--dir", str(directory)])
    main(["read", "--dir", str(directory), "--partial"])


def test_a_refit_without_its_direction_or_with_two_arms_is_refused(tmp_path: Path) -> None:
    directory = reading_directory(tmp_path)
    for path in (directory / "answers").glob("direction-*.csv"):
        path.unlink()
    rows = [row for row in cells_of(directory) if row["arm"] != "direction"]
    write_rows(directory, [tuple(row.values()) for row in rows])

    with pytest.raises(SystemExit, match="adopt the bars first"):
        main(["read", "--dir", str(directory)])

    grown: list[tuple[object, ...]] = []
    write_cell(
        directory,
        grown,
        arm=arm_name("refit", 0.001, 1000),
        encoder="base1",
        events=9,
        seed=1,
        steps=1000,
        lift=0.0,
    )
    write_rows(directory, [tuple(row.values()) for row in rows] + grown)
    with pytest.raises(SystemExit, match="two refit arms"):
        main(["read", "--dir", str(directory), "--partial"])


def test_a_directory_without_a_cell_is_refused(tmp_path: Path) -> None:
    write_rows(tmp_path, [])

    with pytest.raises(SystemExit, match="no cell"):
        main(["read", "--dir", str(tmp_path)])


def test_each_half_holds_the_normal_windows_and_the_events_of_its_classes() -> None:
    answers = [
        predicted("m", 0, 0.0, 0.1),
        predicted("m", 1, 1.0, 0.9),
        predicted("m", 2, 2.0, 0.8),
    ]

    halves = split_by_seen(answers, {1: "class_1", 2: "class_2"}, frozenset({"class_1"}))

    assert [each.window.position for each in halves[SEEN_HALF]] == [0, 1]
    assert [each.window.position for each in halves[UNSEEN_HALF]] == [0, 2]


def test_the_class_rows_read_the_refit_and_the_bars_but_not_the_selection(tmp_path: Path) -> None:
    directory = reading_directory(tmp_path)
    classes = {month: "class_a" if month <= 3 else "class_b" for month in range(1, 7)}

    rows = class_rows(cells_of(directory), directory, classes, frozenset({"class_a"}))

    arms = {str(row[0]) for row in rows}
    assert arms == {"refit-ft@0.001-steps500", "direction", "trees"}
    assert {str(row[3]) for row in rows} == {SEEN_HALF, UNSEEN_HALF}
    assert all(row[4] == 3 for row in rows)
    text = class_table(rows)
    assert "Seen classes (3 events)" in text
    assert "| direction | base1 |" in text


def test_a_plan_names_its_cell() -> None:
    plan = Plan(section="refit", rate=0.0003, steps=4000, encoder="base2", seed=3, events=99)

    assert plan.cell == "refit-ft@0.0003-steps4000-base2-k99-s3"
