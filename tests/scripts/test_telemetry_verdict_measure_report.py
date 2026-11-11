"""The verdict measures over the departure arms: the benchmark's F-score, chance and months."""

import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.taught_fine_tuning_pilot import write_answers
from scripts.telemetry_channel_local_report import (
    LOCAL,
    MAX,
    MEAN,
    NO_SEED,
    LeftOut,
    SideArrays,
    alarm_boundary,
    cell_name,
)
from scripts.telemetry_context_report import REFIT, SELECT
from scripts.telemetry_verdict_measure_report import (
    FReading,
    alarms,
    chance,
    chance_by_month,
    choose_by_f,
    choose_f,
    f_reading,
    main,
    month_draws,
    separated,
    side_answers,
    tally,
    value,
    verdict_value,
)


def predicted(unit: str, position: int, target: float, score: float) -> WindowPrediction:
    return WindowPrediction(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=float(position)),
        target=target,
        predicted=score,
    )


def answers(seed: int, *, months: int = 3, ties: bool = False) -> list[WindowPrediction]:
    """Months of 60 windows, events of 1 to 12 windows, scores around nought, some tied."""
    draw = random.Random(seed)
    out = []
    event = 0
    for month in range(months):
        place = 0
        while place < 60:
            if draw.random() < 0.05:
                event += 1
                length = draw.randint(1, 12)
                for _ in range(min(length, 60 - place)):
                    score = draw.randint(-3, 3) if ties else draw.gauss(0.5, 1.0)
                    out.append(predicted(f"m/200{month}-01", place, float(event), score))
                    place += 1
            else:
                score = draw.randint(-3, 3) if ties else draw.gauss(-1.0, 1.0)
                out.append(predicted(f"m/200{month}-01", place, 0.0, score))
                place += 1
    return out


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ties", [False, True])
def test_the_own_alarms_score_is_the_benchmarks_corrected_event_f(seed: int, ties: bool) -> None:
    rows = answers(seed, ties=ties)
    ranking = AnomalyRanking.of(rows)

    assert value(SideArrays.of(rows), "f_own") == pytest.approx(ranking.corrected_event_fbeta(0.5))
    assert value(SideArrays.of(rows), "found") == round(
        ranking.event_recall_at_false_alarm_share(0.01) * ranking.event_count
    )


@pytest.mark.parametrize("seed", range(10))
def test_the_budgets_score_is_the_corrected_event_f_at_one_per_cent(seed: int) -> None:
    rows = answers(seed, ties=True)
    arrays = SideArrays.of(rows)
    boundary = alarm_boundary(arrays.scores, arrays.events, 0.01)
    shifted = [replace(p, predicted=p.predicted - boundary) for p in rows]

    expected = AnomalyRanking.of(shifted).corrected_event_fbeta(0.5)

    assert value(arrays, "f_budget") == pytest.approx(expected)
    assert int(alarms(arrays, "f_budget").sum()) == int((arrays.scores > boundary).sum())


def test_a_month_drawn_twice_counts_as_that_month_repeated() -> None:
    rows = answers(4)
    arrays = SideArrays.of(rows)
    again = [
        replace(p, window=replace(p.window, unit=UnitKey("m/2009-01")))
        for p in rows
        if str(p.window.unit) == "m/2001-01"
    ]
    shifted = [replace(p, target=p.target + 1000 if p.target else 0.0) for p in again]
    doubled = SideArrays.of(rows + shifted)

    for name in ("found", "f_own", "f_budget"):
        flags = alarms(arrays, name)
        weighted = tally(arrays, flags).measure(name, np.asarray([1.0, 2.0, 1.0]))
        doubled_flags = (
            alarms(doubled, name)
            if name == "f_own"
            else doubled.scores > alarm_boundary(arrays.scores, arrays.events, 0.01)
        )
        assert weighted == pytest.approx(tally(doubled, doubled_flags).measure(name))


def test_a_run_of_alarms_touching_an_event_is_no_false_alarm() -> None:
    rows = [
        predicted("m/2000-01", place, target, score)
        for place, (target, score) in enumerate(
            [(0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, 1.0), (0.0, -1.0)]
        )
    ]

    counted = tally(SideArrays.of(rows), alarms(SideArrays.of(rows), "f_own"))

    assert (int(counted.found.sum()), int(counted.false_runs.sum())) == (1, 1)
    assert (int(counted.normal.sum()), int(counted.quiet.sum())) == (6, 2)


def test_chance_is_the_same_for_the_same_scores_and_lower_than_a_detector_that_finds() -> None:
    arrays = SideArrays.of(answers(2))
    finds = replace(arrays, scores=np.where(arrays.events > 0, 5.0, arrays.scores))

    expected = chance(
        {(MAX, "a"): arrays, (MEAN, "a"): arrays, (MAX, "b"): finds}, rotations=30, seed=1
    )

    assert expected[(MAX, "a")] == expected[(MEAN, "a")]
    assert expected[(MAX, "b")]["found"] < value(finds, "found")


def test_the_paired_draws_tell_apart_a_detector_that_finds_from_one_that_does_not() -> None:
    arrays = SideArrays.of(answers(3, months=8))
    finds = replace(arrays, scores=np.where(arrays.events > 0, 5.0, arrays.scores))
    blind = replace(arrays, scores=np.where(arrays.events > 0, -5.0, arrays.scores))
    good, bad = (tally(a, alarms(a, "found")) for a in (finds, blind))
    weights = month_draws([good], 300, 1)

    assert separated(good, bad, "found", weights)
    assert not separated(good, good, "found", weights)
    assert weights.sum(axis=1).tolist() == [8.0] * 300


def a_run(directory: Path, sides: dict[str, list[str]]) -> None:
    (directory / "conditions.json").write_text(
        json.dumps({"features": ["hour", "bins"], "spans": [6]})
    )
    (directory / "answers").mkdir()
    for side, arms in sides.items():
        (directory / f"truncated-{side}.csv").write_text("unit,position,ends_at,target\n")
        for arm in arms:
            for detector, shift in ((MAX, 0.0), (MEAN, 0.7)):
                name = cell_name(side, LOCAL, detector, arm, NO_SEED)
                rows = [
                    replace(p, predicted=p.predicted + shift * p.target)
                    for p in answers(1, months=6)
                ]
                write_answers(directory / "answers" / f"{name}.csv", rows)


def test_the_report_measures_both_sides_and_reads_its_tables_from_the_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a_run(tmp_path, {SELECT: ["hour-6h", "bins-6h"], REFIT: ["hour-6h"]})
    out = tmp_path / "out"

    main(
        [
            "measure",
            "--dir",
            str(tmp_path),
            "--out",
            str(out),
            "--rotations",
            "5",
            "--resamples",
            "50",
        ]
    )

    with (out / "arms.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == (4 + 2) * 3
    text = capsys.readouterr().out
    assert "| f_own | select | 4 |" in text
    assert "(2) |" in text
    assert (out / "tables.md").read_text() == text


def test_a_section_with_no_arm_or_with_other_windows_is_refused(tmp_path: Path) -> None:
    a_run(tmp_path, {SELECT: ["hour-6h"]})
    with pytest.raises(SystemExit, match="no no-model arm"):
        side_answers(tmp_path, REFIT)
    name = cell_name(SELECT, LOCAL, MAX, "hour-6h", NO_SEED)
    write_answers(tmp_path / "answers" / f"{name}.csv", answers(1, months=6)[1:])
    with pytest.raises(SystemExit, match="other windows"):
        side_answers(tmp_path, SELECT)


def test_the_excess_by_month_sums_to_found_less_chance_and_weighs_months() -> None:
    arrays = SideArrays.of(answers(5, months=4))
    counted = tally(arrays, alarms(arrays, "found"))
    by_month = chance_by_month({(MAX, "a"): arrays}, rotations=40, seed=1)[(MAX, "a")]

    assert verdict_value("excess", counted, by_month, None) == pytest.approx(
        counted.found.sum() - by_month.sum()
    )
    assert by_month.sum() == pytest.approx(
        chance({(MAX, "a"): arrays}, rotations=40, seed=1)[(MAX, "a")]["found"]
    )
    doubled = np.asarray([2.0, 1.0, 1.0, 0.0])
    assert verdict_value("excess", counted, by_month, doubled) == pytest.approx(
        float(doubled @ (counted.found - by_month))
    )
    assert verdict_value("f_own", counted, by_month, None) == counted.measure("f_own")


def test_the_power_report_gives_each_pair_a_least_gain_per_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a_run(tmp_path, {REFIT: ["hour-6h", "bins-6h"]})
    out = tmp_path / "out"

    main(
        [
            "power",
            "--dir",
            str(tmp_path),
            "--out",
            str(out),
            "--rotations",
            "5",
            "--resamples",
            "40",
        ]
    )

    with (out / "power.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 * 2
    for row in rows:
        assert float(row["least_gain"]) == pytest.approx(3.083 * float(row["sd"]), abs=1e-3)
    assert "| excess | 6 |" in capsys.readouterr().out


def f_of(detector: str, arm: str, f_own: float, found: int) -> FReading:
    return FReading(
        detector=detector,
        arm=arm,
        events=20,
        f_own=f_own,
        found=found,
        chance=3.0,
        excess=found - 3.0,
        false_runs=10,
        flagged=0.01,
    )


def test_the_choice_by_f_takes_the_highest_then_the_most_found_then_the_plainest() -> None:
    quiet = f_of(MAX, "bins-72h", 0.5, 8)
    assert choose_by_f([f_of(MEAN, "hour-6h", 0.2, 15), quiet]) == quiet
    finds = f_of(MEAN, "bins-72h", 0.5, 9)
    assert choose_by_f([quiet, finds]) == finds
    tied = [
        f_of(MEAN, "hour-6h", 0.5, 9),
        f_of(MAX, "bins-6h", 0.5, 9),
        f_of(MAX, "hour-6h", 0.5, 9),
    ]
    assert choose_by_f(tied) == tied[2]
    with pytest.raises(ValueError, match="no arm"):
        choose_by_f([])


def test_an_f_reading_splits_the_score_into_what_makes_it() -> None:
    rows = [
        predicted("m/2000-01", place, target, score)
        for place, (target, score) in enumerate(
            [(0.0, 1.0), (1.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, -1.0)] + [(0.0, -2.0)] * 95
        )
    ]
    arrays = SideArrays.of(rows)

    reading = f_reading((MAX, "hour-6h"), arrays, 0.25)

    assert (reading.events, reading.found, reading.false_runs) == (1, 1, 1)
    assert reading.flagged == pytest.approx(2 / 99)
    assert reading.f_own == pytest.approx(AnomalyRanking.of(rows).corrected_event_fbeta(0.5))


def test_the_choice_by_f_is_written_from_its_csv_and_leaves_out_what_is_named(
    tmp_path: Path,
) -> None:
    a_run(tmp_path, {SELECT: ["hour-6h", "bins-6h"]})
    out = tmp_path / "out"
    left_out = LeftOut(classes={"class_x": [1]}, overlaps=lambda w: ())

    text = choose_f(tmp_path, out, section=SELECT, left_out=left_out, rotations=5)
    choose_f(tmp_path, out, section=SELECT, rotations=5, trial=3)

    with (out / "f-select-without-class_x.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert sum(int(r["chosen"]) for r in rows) == 1
    chosen = json.loads((out / "f-select-without-class_x-chosen.json").read_text())
    assert (chosen["left_out"], chosen["trial"]) == (["class_x"], None)
    assert (out / "f-select-without-class_x.md").read_text() == text + "\n"
    assert (out / "f-select-trial-3.csv").exists()


def test_a_run_of_alarms_ends_where_its_month_ends_even_if_places_run_on() -> None:
    # Two months whose places continue, each ending and starting with a flagged normal window.
    rows = [
        predicted("m/2000-01", 0, 0.0, -1.0),
        predicted("m/2000-01", 1, 0.0, 1.0),
        predicted("m/2000-02", 2, 0.0, 1.0),
        predicted("m/2000-02", 3, 1.0, -1.0),
    ]
    arrays = SideArrays.of(rows)

    counted = tally(arrays, alarms(arrays, "f_own"))

    assert int(counted.false_runs.sum()) == 2
    assert counted.false_runs.tolist() == [1, 1]
