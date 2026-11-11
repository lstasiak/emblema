"""The first mission's verdict: the excess against the best departure arm, the F-score beside."""

import argparse
import csv
import random
from collections.abc import Sequence
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
    TAUGHT,
    SideArrays,
    cell_name,
    month_slices,
    rotation_order,
)
from scripts.telemetry_context_report import REFIT, TREES
from scripts.telemetry_verdict_measure_report import (
    FLOOR,
    alarms,
    ceiling,
    chance_by_month,
    excess,
    family,
    main,
    month_draws,
    oracle_of,
    parse_taught,
    tally,
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
def test_the_counts_are_the_benchmarks_f_score_and_the_harness_recall(
    seed: int, ties: bool
) -> None:
    rows = answers(seed, ties=ties)
    ranking = AnomalyRanking.of(rows)
    arrays = SideArrays.of(rows)

    assert tally(arrays, alarms(arrays, own=True)).f_score() == pytest.approx(
        ranking.corrected_event_fbeta(0.5)
    )
    assert int(tally(arrays, alarms(arrays)).found.sum()) == round(
        ranking.event_recall_at_false_alarm_share(0.01) * ranking.event_count
    )


def test_a_run_of_alarms_touching_an_event_is_no_false_alarm() -> None:
    rows = [
        predicted("m/2000-01", place, target, score)
        for place, (target, score) in enumerate(
            [(0.0, 1.0), (1.0, 1.0), (0.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, 1.0), (0.0, -1.0)]
        )
    ]
    arrays = SideArrays.of(rows)

    counted = tally(arrays, alarms(arrays, own=True))

    assert (int(counted.found.sum()), int(counted.false_runs.sum())) == (1, 1)
    assert (int(counted.normal.sum()), int(counted.quiet.sum())) == (6, 2)


def test_a_run_of_alarms_ends_where_its_month_ends_even_if_places_run_on() -> None:
    # Two months whose places continue, each ending and starting with a flagged normal window.
    rows = [
        predicted("m/2000-01", 0, 0.0, -1.0),
        predicted("m/2000-01", 1, 0.0, 1.0),
        predicted("m/2000-02", 2, 0.0, 1.0),
        predicted("m/2000-02", 3, 1.0, -1.0),
    ]
    arrays = SideArrays.of(rows)

    assert tally(arrays, alarms(arrays, own=True)).false_runs.tolist() == [1, 1]


def test_months_are_drawn_within_their_stratum_and_the_excess_weighs_them() -> None:
    events = np.asarray([2, 0, 1, 0, 0], dtype=np.int64)

    weights = month_draws(events, 200, 1)

    assert weights.sum(axis=1).tolist() == [5.0] * 200
    assert (weights[:, [0, 2]].sum(axis=1) == 2).all()
    found = np.asarray([2, 0, 1, 0, 0], dtype=np.int64)
    chance = np.asarray([0.5, 0.1, 0.2, 0.0, 0.0])
    assert excess(found, chance) == pytest.approx(2.2)
    assert excess(found, chance, np.asarray([2.0, 1.0, 0.0, 1.0, 1.0])) == pytest.approx(2.9)


def test_chance_is_shared_and_below_a_detector_that_finds() -> None:
    arrays = SideArrays.of(answers(2))
    finds = replace(arrays, scores=np.where(arrays.events > 0, 5.0, arrays.scores))

    expected = chance_by_month(
        {(MAX, "a"): arrays, (MEAN, "a"): arrays, (MAX, "b"): finds}, rotations=30, seed=1
    )

    assert expected[(MAX, "a")].tolist() == expected[(MEAN, "a")].tolist()
    assert expected[(MAX, "b")].sum() < tally(finds, alarms(finds)).found.sum()


def test_a_rotation_keeps_the_share_the_own_alarm_flags_in_each_month() -> None:
    arrays = SideArrays.of(answers(6, months=4))
    finds = replace(arrays, scores=np.where(arrays.events > 0, 5.0, arrays.scores))
    slices = month_slices(finds.units)
    again = finds.turned(rotation_order(slices, np.random.default_rng(3)))

    assert (
        np.bincount(finds.units[alarms(again, own=True)], minlength=4).tolist()
        == np.bincount(finds.units[alarms(finds, own=True)], minlength=4).tolist()
    )
    own = chance_by_month({(MAX, "a"): finds}, rotations=30, seed=1, own=True)[(MAX, "a")]
    draws = np.random.default_rng(1)
    expected = np.zeros(4)
    for _ in range(30):
        turned = finds.turned(rotation_order(slices, draws))
        expected += tally(turned, alarms(turned, own=True)).found
    assert own.tolist() == pytest.approx((expected / 30).tolist())


def a_run(directory: Path, arms: Sequence[str] = ("hour-6h", "bins-6h")) -> None:
    (directory / "conditions.json").write_text('{"features": ["hour", "bins"], "spans": [6]}')
    (directory / "answers").mkdir()
    (directory / f"truncated-{REFIT}.csv").write_text("unit,position,ends_at,target\n")
    for arm in arms:
        for detector, shift in ((MAX, 0.0), (MEAN, 0.7)):
            name = cell_name(REFIT, LOCAL, detector, arm, NO_SEED)
            rows = [
                replace(p, predicted=p.predicted + shift * p.target) for p in answers(1, months=6)
            ]
            write_answers(directory / "answers" / f"{name}.csv", rows)


def taught_cells(directory: Path, arm: str, seeds: Sequence[int], lift: float) -> None:
    """Trees taught the events: the arms' windows, event windows lifted more under each seed."""
    for seed in seeds:
        name = cell_name(REFIT, TAUGHT, TREES, arm, seed)
        rows = [
            replace(p, predicted=p.predicted + (lift + 0.1 * seed) * (p.target > 0))
            for p in answers(1, months=6)
        ]
        write_answers(directory / "answers" / f"{name}.csv", rows)


def test_each_arm_stands_against_the_best_of_the_rest(tmp_path: Path) -> None:
    a_run(tmp_path)

    rows = family(tmp_path, tmp_path / "out", rotations=4, resamples=30)

    assert len(rows) == 4
    values = {str(r["arm"]): float(str(r["value"])) for r in rows}
    for row in rows:
        rest = [v for arm, v in values.items() if arm != row["arm"]]
        assert float(str(row["best_of_rest"])) == pytest.approx(max(rest), abs=1e-4)
    sds = sorted(float(str(r["sd"])) for r in rows)
    assert float(str(rows[0]["least_gain"])) == pytest.approx(
        3.083 * (sds[1] + sds[2]) / 2, abs=1e-3
    )


def test_a_grid_arm_without_answers_is_refused(tmp_path: Path) -> None:
    a_run(tmp_path, arms=("hour-6h",))
    with pytest.raises(SystemExit, match="refit-local-max-bins-6h is missing"):
        family(tmp_path, tmp_path / "out", rotations=2, resamples=5)


def test_the_oracle_flags_every_event_window_and_no_other() -> None:
    arrays = SideArrays.of(answers(2, months=4))
    oracle = oracle_of(arrays)

    assert tally(oracle, alarms(oracle, own=True)).f_score() == pytest.approx(1.0)
    assert int(tally(oracle, alarms(oracle)).found.sum()) == len(arrays.event_ids)


def test_the_ceiling_reads_each_reference_against_the_best_arm_as_a_candidate(
    tmp_path: Path,
) -> None:
    a_run(tmp_path)
    taught_cells(tmp_path, "hour-6h", (1, 2, 3), lift=3.0)

    rows = ceiling(
        tmp_path,
        tmp_path,
        tmp_path / "out",
        references=[(TREES, "hour-6h")],
        rotations=5,
        resamples=60,
    )

    readings = {(r["reference"], r["seeds"]): r for r in rows}
    assert len(rows) == 1 + 3 + 1
    seeds = [float(str(readings[("trees hour-6h", f"s{s}")]["value"])) for s in (1, 2, 3)]
    mean = float(str(readings[("trees hour-6h", "all")]["value"]))
    assert mean == pytest.approx(float(np.mean(seeds)), abs=1e-3)
    for row in rows:
        difference = float(str(row["difference"]))
        assert row["confirmed"] == int(float(str(row["low"])) > 0 and difference >= FLOOR)
        assert difference == pytest.approx(
            float(str(row["value"])) - float(str(row["bar"])), abs=2e-4
        )
    held = family(tmp_path, tmp_path / "family", rotations=5, resamples=10)
    assert {str(r["bar"]) for r in rows} == {f"{max(float(str(r['value'])) for r in held):.4f}"}
    # Turned, the oracle's flags outnumber the budget and tie at the alarm, so its chance is
    # nought and its excess every event.
    events = len(SideArrays.of(answers(1, months=6)).event_ids)
    assert float(str(readings[("oracle", "")]["value"])) == events
    assert float(str(readings[("oracle", "")]["f_own"])) == 1.0


def test_the_ceiling_refuses_a_missing_seed_or_other_windows(tmp_path: Path) -> None:
    a_run(tmp_path)
    taught_cells(tmp_path, "hour-6h", (1, 2), lift=1.0)

    def read() -> None:
        ceiling(
            tmp_path,
            tmp_path,
            tmp_path / "out",
            references=[(TREES, "hour-6h")],
            rotations=2,
            resamples=5,
        )

    with pytest.raises(SystemExit, match="refit-taught-trees-hour-6h-s3 is missing"):
        read()
    name = cell_name(REFIT, TAUGHT, TREES, "hour-6h", 3)
    write_answers(tmp_path / "answers" / f"{name}.csv", answers(1, months=6)[1:])
    with pytest.raises(SystemExit, match="other windows or events than the no-model arms"):
        read()


def test_a_reference_is_paired_with_the_bar_in_every_draw(tmp_path: Path) -> None:
    a_run(tmp_path)
    # A reference scoring as the strongest arm ties the bar in every draw: an interval of nought
    # only if each draw compares like with like.
    best = max(
        family(tmp_path, tmp_path / "family", rotations=4, resamples=10),
        key=lambda r: float(str(r["value"])),
    )
    detector, arm = str(best["arm"]).split()
    source = tmp_path / "answers" / f"{cell_name(REFIT, LOCAL, detector, arm, NO_SEED)}.csv"
    for seed in (1, 2, 3):
        name = cell_name(REFIT, TAUGHT, TREES, "hour-6h", seed)
        (tmp_path / "answers" / f"{name}.csv").write_text(source.read_text())

    rows = ceiling(
        tmp_path,
        tmp_path,
        tmp_path / "out",
        references=[(TREES, "hour-6h")],
        rotations=4,
        resamples=40,
    )

    tied = [r for r in rows if r["reference"] == "trees hour-6h"]
    assert {(r["low"], r["high"], r["difference"]) for r in tied} == {
        ("0.0000", "0.0000", "0.0000")
    }


def test_a_ceiling_trial_turns_the_references_with_the_arms(tmp_path: Path) -> None:
    a_run(tmp_path)
    taught_cells(tmp_path, "hour-6h", (1,), lift=3.0)

    def found(trial: int | None, reference: str) -> float:
        rows = ceiling(
            tmp_path,
            tmp_path,
            tmp_path / "out",
            references=[(TREES, "hour-6h")],
            seeds=(1,),
            rotations=3,
            resamples=20,
            trial=trial,
        )
        return next(float(str(r["found"])) for r in rows if r["reference"] == reference)

    events = len(SideArrays.of(answers(1, months=6)).event_ids)
    assert found(None, "oracle") == events
    assert found(1, "oracle") == 0
    assert found(1, "trees hour-6h") < found(None, "trees hour-6h")


def test_main_routes_both_commands_and_a_trial_writes_apart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a_run(tmp_path)
    taught_cells(tmp_path, "hour-6h", (1,), lift=2.0)
    out = tmp_path / "out"

    main(["family", "--dir", str(tmp_path), "--out", str(out), "--rotations", "3"])
    assert "| Arm | Value |" in capsys.readouterr().out
    main(
        [
            "ceiling",
            *("--dir", str(tmp_path), "--taught", str(tmp_path), "--out", str(out)),
            *("--taught-arm", "trees:hour-6h", "--seeds", "1"),
            *("--rotations", "3", "--resamples", "20", "--trial", "2"),
        ]
    )

    assert "| Reference | Seeds |" in capsys.readouterr().out
    assert (out / "family.md").exists()
    assert (out / "ceiling-trial-2.md").exists()
    assert not (out / "ceiling.csv").exists()
    with (out / "ceiling-trial-2.csv").open(newline="") as handle:
        # One seed reads as itself, beside the oracle.
        assert len(list(csv.DictReader(handle))) == 2


def test_a_taught_arm_is_named_detector_then_arm() -> None:
    assert parse_taught("trees:bins-24h") == (TREES, "bins-24h")
    with pytest.raises(argparse.ArgumentTypeError, match="<detector>:<arm>"):
        parse_taught("max:bins-24h")
