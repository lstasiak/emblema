"""The context layout against the layout over concatenated tokens, the choice and the reading."""

import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.taught_fine_tuning_pilot import write_answers
from scripts.telemetry_context_report import (
    BASE_BINS,
    CELL_COLUMNS,
    CELLS,
    CONTROL,
    DIRECTION,
    POOLED,
    REFIT,
    SELECT,
    SMOOTHED_MAX,
    SMOOTHED_MEAN,
    STATES,
    WHITENED,
    Layout,
    PublishedCells,
    Work,
    bins_of,
    cell_name,
    check_cell,
    choice_table,
    choose,
    choose_from,
    context_cells,
    context_row,
    corrected,
    events_found,
    hours_of,
    kept,
    laid_context,
    lane_of,
    lane_table,
    main,
    missing_refits,
    month_of_hour,
    pooled_rows,
    read_tables,
    read_truncated,
    refit_work,
    refuse_other_conditions,
    select_work,
    smoothed,
    smoothed_cells,
    span_name,
    truncated_windows,
    write_truncated,
)

CHANNELS = (1, 2, 3, 4)
QUICK = PairedUnitBootstrap(resamples=400, seed=1)


def hour_window(draw: random.Random, *, channels: tuple[int, ...], readings: int) -> TokenWindow:
    """One published window of an hour: readings of the given channels at random instants."""
    tokens = [
        Token(
            channel_id=draw.choice(channels),
            value=round(draw.uniform(-3.0, 3.0), 3),
            time=draw.random(),
            gap=0.0,
        )
        for _ in range(readings)
    ]
    return TokenWindow.of(tokens)


def concatenated(hours: list[TokenWindow | None]) -> TokenWindow:
    """The hours as one window, each hour's times placed in its share of the whole."""
    count = len(hours)
    tokens = [
        Token(channel_id=t.channel_id, value=t.value, time=(at + t.time) / count, gap=0.0)
        for at, hour in enumerate(hours)
        if hour is not None
        for t in hour
    ]
    return TokenWindow.of(tokens)


def published_cells(hours: list[TokenWindow | None]) -> np.ndarray:
    """``[rows, hours * BASE_BINS, features]``: each hour laid over the corpus's channels."""
    layout = BinnedTokens(BASE_BINS, channels=CHANNELS)
    laid = []
    for hour in hours:
        if hour is None:
            laid.append(np.zeros((len(CHANNELS), BASE_BINS, BinnedTokens.N_FEATURES)))
        else:
            cells = layout.lay([hour])[0].features
            laid.append(cells.reshape(BASE_BINS, len(CHANNELS), -1).transpose(1, 0, 2))
    return np.concatenate(laid, axis=1)


@pytest.mark.parametrize(("hours", "bins"), [(1, 6), (6, 6), (6, 36), (24, 24), (4, 8), (3, 2)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_context_cells_are_what_the_layout_makes_of_the_concatenated_tokens(
    hours: int, bins: int, seed: int
) -> None:
    draw = random.Random(seed)
    windows: list[TokenWindow | None] = [
        None
        if draw.random() < 0.2
        else hour_window(
            draw, channels=draw.choice([(1, 2), (2, 3, 4), CHANNELS]), readings=draw.randint(1, 30)
        )
        for _ in range(hours)
    ]
    if all(w is None for w in windows):
        windows[-1] = hour_window(draw, channels=CHANNELS, readings=5)
    whole = concatenated(windows)

    derived = context_cells(published_cells(windows), bins)
    over_corpus = BinnedTokens(bins, channels=CHANNELS).lay([whole])[0]
    own = BinnedTokens(bins).lay([whole])[0]

    assert derived.shape == (len(CHANNELS), bins, BinnedTokens.N_FEATURES)
    expected = over_corpus.features.reshape(bins, len(CHANNELS), -1).transpose(1, 0, 2)
    np.testing.assert_allclose(derived, expected, atol=1e-9)
    rows = np.asarray(CHANNELS, dtype=np.int64)
    laid_corpus = laid_context(derived, rows, own_channels=False)
    np.testing.assert_allclose(laid_corpus.features, over_corpus.features, atol=1e-9)
    assert laid_corpus.channel_ids.tolist() == over_corpus.channel_ids.tolist()
    np.testing.assert_allclose(laid_corpus.times, over_corpus.times)
    laid_own = laid_context(derived, rows, own_channels=True)
    np.testing.assert_allclose(laid_own.features, own.features, atol=1e-9)
    assert laid_own.channel_ids.tolist() == own.channel_ids.tolist()
    assert not laid_own.timeless.any()


def test_cells_that_do_not_cut_into_the_bins_are_refused() -> None:
    with pytest.raises(ValueError, match="do not cut"):
        context_cells(np.zeros((2, 12, BinnedTokens.N_FEATURES)), 5)
    with pytest.raises(ValueError, match="do not cut"):
        Layout(1, 4)
    with pytest.raises(ValueError, match="hours and bins"):
        Layout(0, 6)


def test_a_layout_is_named_by_its_hours_and_bins() -> None:
    assert Layout.parse("24x6") == Layout(24, 6)
    assert Layout(6, 36).name == "6x36"
    assert Layout(24, 6).cells_per_bin == 24
    assert Layout(1, BASE_BINS) == CONTROL
    assert (hours_of("24x6"), bins_of("24x6")) == (24, 6)
    assert (hours_of(span_name(72)), bins_of("72h")) == (72, BASE_BINS)
    assert sorted([Layout(24, 6), Layout(6, 36), Layout(6, 6)]) == [
        Layout(6, 6),
        Layout(6, 36),
        Layout(24, 6),
    ]
    with pytest.raises(ValueError, match="<hours>x<bins>"):
        Layout.parse("24")


def test_the_published_cells_gather_the_hours_before_a_window_and_nothing_across_a_hole() -> None:
    cells = np.zeros((4, 1, BASE_BINS, BinnedTokens.N_FEATURES))
    for window in range(4):
        cells[window, 0, :, 0] = 1.0
        cells[window, 0, :, 1] = np.log1p(1.0)
        cells[window, 0, :, 2] = float(window + 1)
    # Windows at hours 10, 11, 12 and 15: the hours 13 and 14 were never loaded, so the window
    # at 15 sees nothing of them and, four hours back, only the window at 12.
    published = PublishedCells(cells, np.array([7]), [10.0, 11.0, 12.0, 15.0])

    derived = published.context([2, 3], Layout(3, 3))

    assert derived.shape == (2, 1, 3, BinnedTokens.N_FEATURES)
    np.testing.assert_allclose(derived[0, 0, :, 2], [1.0, 2.0, 3.0])
    assert derived[1, 0, :, 0].tolist() == [0.0, 0.0, 1.0]
    assert published.previous([3], 4).tolist() == [[2, -1, -1]]
    assert published.previous([2], 3).tolist() == [[0, 1]]
    assert published.previous([0], 2).tolist() == [[-1]]


def test_published_cells_refuse_windows_off_the_hour_or_sharing_one() -> None:
    cells = np.zeros((2, 1, BASE_BINS, BinnedTokens.N_FEATURES))
    with pytest.raises(ValueError, match="whole hours"):
        PublishedCells(cells, np.array([1]), [1.0, 1.5])
    # A few seconds off the hour is read at the nearest whole hour.
    assert PublishedCells(cells, np.array([1]), [1.0044, 2.0044]).previous([1], 2).tolist() == [[0]]
    with pytest.raises(ValueError, match="end at one hour"):
        PublishedCells(cells, np.array([1]), [1.0, 1.0])


def test_pooled_rows_put_the_mean_of_the_loaded_hours_before_beside_the_hour() -> None:
    cells = np.zeros((3, 1, BASE_BINS, BinnedTokens.N_FEATURES))
    published = PublishedCells(cells, np.array([1]), [5.0, 6.0, 9.0])
    states = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 2.0]])

    rows = pooled_rows(states, published, [0, 1, 2], hours=3)

    np.testing.assert_allclose(rows[0], [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(rows[1], [3.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(rows[2], [5.0, 2.0, 5.0, 2.0])
    rows = pooled_rows(states, published, [2], hours=5)
    np.testing.assert_allclose(rows[0], [5.0, 2.0, 2.0, 0.0])


def predicted(
    unit: str, position: int, ends_at: float, target: float, score: float
) -> WindowPrediction:
    return WindowPrediction(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at),
        target=target,
        predicted=score,
    )


def test_smoothing_reads_the_scored_hours_ending_at_each_window() -> None:
    answers = [
        predicted("m/1", 0, 10.0, 0.0, 1.0),
        predicted("m/1", 1, 11.0, 0.0, 5.0),
        predicted("m/1", 2, 12.0, 1.0, 2.0),
        predicted("m/2", 3, 20.0, 0.0, 4.0),
    ]

    largest = smoothed(answers, 2, how=SMOOTHED_MAX)
    mean = smoothed(answers, 3, how=SMOOTHED_MEAN)

    assert [p.predicted for p in largest] == [1.0, 5.0, 5.0, 4.0]
    assert [p.predicted for p in mean] == [1.0, 3.0, pytest.approx(8.0 / 3.0), 4.0]
    assert [p.target for p in mean] == [0.0, 0.0, 1.0, 0.0]


def test_the_selection_runs_the_widest_layouts_first_and_the_lanes_share_the_units() -> None:
    work = select_work([Layout(1, 6), Layout(24, 6), Layout(6, 36)], [6], [1, 2])

    assert [w.arm for w in work[:2]] == ["6x36", "6x36"]
    assert [w.family for w in work if w.arm == "24x6"] == [STATES, STATES, CELLS, CELLS]
    assert [w.detectors for w in work if w.family == CELLS and w.arm == "24x6"] == [
        ("farthest", "trees"),
        ("trees",),
    ]
    assert [w.family for w in work if w.arm == "6x36"] == [STATES, STATES]
    assert work[-1] == Work(
        section=SELECT, family=POOLED, arm="6h", seed=2, detectors=(WHITENED, DIRECTION)
    )
    assert work[0].cells() == ["select-states-6x36-whitened-s1", "select-states-6x36-direction-s1"]
    assert sorted(
        w.cells()[0] for lane in ("1/3", "2/3", "3/3") for w in lane_of(work, lane)
    ) == sorted(w.cells()[0] for w in work)
    assert lane_of(work, "2/3") == work[1::3]
    assert lane_of(work, None) == work


@pytest.mark.parametrize("lane", ["0/3", "4/3", "a/b", "2", "1/2/3"])
def test_a_lane_that_is_not_i_of_n_is_refused(lane: str) -> None:
    with pytest.raises(SystemExit, match="i/n"):
        lane_of([], lane)


def test_each_lane_records_its_cells_in_its_own_table(tmp_path: Path) -> None:
    assert lane_table(tmp_path, None) == tmp_path / "cells.csv"
    assert lane_table(tmp_path, "2/3") == tmp_path / "cells-2-of-3.csv"


def test_the_choice_takes_the_shortest_arm_within_the_margin_of_the_best() -> None:
    means = {"1x6": 0.20, "6x6": 0.26, "24x6": 0.265, "6x36": 0.25, "24x24": 0.24}

    choice = choose(STATES, DIRECTION, means)

    assert choice.arm == "6x6"
    assert choice.best_context == "24x6"
    assert choice.chose_context
    bins = choose(STATES, DIRECTION, {"1x6": 0.10, "6x36": 0.30, "6x6": 0.295, "24x6": 0.20})
    assert bins.arm == "6x6"
    control = choose(STATES, WHITENED, {"1x6": 0.10, "6x6": 0.105, "24x6": 0.09})
    assert control.arm == "1x6"
    assert not control.chose_context
    assert control.best_context == "6x6"
    with pytest.raises(ValueError, match="control"):
        choose(STATES, WHITENED, {"6x6": 0.1})
    assert "Chosen" in choice_table([choice, control])


def write_cell(
    directory: Path,
    cell: str,
    section: str,
    family: str,
    arm: str,
    detector: str,
    seed: int,
    answers: list[WindowPrediction],
) -> None:
    table_path = directory / "cells.csv"
    new = not table_path.exists()
    with table_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(CELL_COLUMNS)
        writer.writerow((cell, section, family, arm, detector, seed, 3, 10, len(answers), "1.0"))
    (directory / "answers").mkdir(exist_ok=True)
    write_answers(directory / "answers" / f"{cell}.csv", answers)


def answers_scoring(events_high: bool, seed: int) -> list[WindowPrediction]:
    """Four months of five windows; an event in each; scores that find it or do not."""
    draw = random.Random(seed)
    out = []
    for month in range(4):
        for place in range(5):
            target = 1.0 + month if place == 2 else 0.0
            score = draw.random() + (2.0 if events_high and target else 0.0)
            out.append(
                predicted(
                    f"m/200{month}-01",
                    month * 5 + place,
                    float(100 + month * 5 + place),
                    target,
                    score,
                )
            )
    return out


def test_the_choice_from_the_selection_cells_and_the_reading_on_the_refit_cells(
    tmp_path: Path,
) -> None:
    conditions = {"arms": ["1x6", "6x6"], "spans": [6], "seeds": [1, 2], "select_from": "2004-01"}
    (tmp_path / "conditions.json").write_text(json.dumps(conditions))
    for section in (SELECT, REFIT):
        for seed in (1, 2):
            for family, arm, detectors, good in (
                (STATES, "1x6", (WHITENED, DIRECTION), False),
                (STATES, "6x6", (WHITENED, DIRECTION), True),
                (CELLS, "1x6", ("farthest", "trees"), False),
                (CELLS, "6x6", ("farthest", "trees"), False),
                (POOLED, "6h", (WHITENED, DIRECTION), False),
                (SMOOTHED_MAX, "6h", (WHITENED, DIRECTION), False),
                (SMOOTHED_MEAN, "6h", (WHITENED, DIRECTION), False),
            ):
                for detector in detectors:
                    write_cell(
                        tmp_path,
                        cell_name(section, family, arm, detector, seed),
                        section,
                        family,
                        arm,
                        detector,
                        seed,
                        answers_scoring(good, seed),
                    )

    choices = choose_from(tmp_path)

    by_key = {(c.family, c.detector): c for c in choices}
    assert by_key[(STATES, DIRECTION)].arm == "6x6"
    assert by_key[(POOLED, WHITENED)].arm == "1x6"
    assert by_key[(CELLS, "trees")].arm == "1x6"
    assert json.loads((tmp_path / "chosen.json").read_text())[0]["family"] == CELLS
    work = refit_work(tmp_path, [1, 2])
    assert {(w.family, w.arm) for w in work} == {
        (STATES, "6x6"),
        (STATES, "1x6"),
        (CELLS, "6x6"),
        (CELLS, "1x6"),
        (POOLED, "6h"),
    }
    assert all(w.section == REFIT for w in work)
    assert {w.detectors for w in work if w.arm == "1x6"} == {
        (WHITENED, DIRECTION),
        ("farthest", "trees"),
        ("trees",),
    }
    assert missing_refits(tmp_path, conditions) == []

    text = read_tables(tmp_path)

    assert (tmp_path / "context.md").exists()
    rows = [line for line in text.splitlines() if line.startswith("| states | direction | 6x6 |")]
    assert len(rows) == 1
    assert "clearly above" in rows[0]
    assert "passes" in rows[0]
    assert "| pooled | whitened | 6h |" in text
    assert "| beside |" in text
    assert "| Family | Detector | Control per seed |" in text
    assert "Only by control" in text
    assert "| refit-states-6x6-direction-s1 |" in text


def test_an_arm_is_refitted_only_under_the_detectors_that_chose_it(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="choose first"):
        refit_work(tmp_path, [1])
    choice = {"best_context": "6x6", "means": {}}
    (tmp_path / "chosen.json").write_text(
        json.dumps(
            [
                {"family": STATES, "detector": WHITENED, "arm": "24x6", **choice},
                {"family": STATES, "detector": DIRECTION, "arm": "1x6", **choice},
                {"family": POOLED, "detector": DIRECTION, "arm": "6h", "best_context": "6h"},
            ]
        )
    )

    work = refit_work(tmp_path, [1])

    by_arm = {(w.family, w.arm): w.detectors for w in work}
    assert by_arm[(STATES, "24x6")] == (WHITENED,)
    assert by_arm[(STATES, "6x6")] == (WHITENED, DIRECTION)
    assert by_arm[(STATES, "1x6")] == (WHITENED, DIRECTION)
    assert by_arm[(POOLED, "6h")] == (DIRECTION,)
    assert [w.arm for w in work if w.family == STATES] == ["24x6", "6x6", "1x6"]
    assert smoothed_cells(REFIT, WHITENED, 1, [6, 24]) == [
        "refit-smoothed-max-6h-whitened-s1",
        "refit-smoothed-max-24h-whitened-s1",
        "refit-smoothed-mean-6h-whitened-s1",
        "refit-smoothed-mean-24h-whitened-s1",
    ]
    with pytest.raises(ValueError, match="smoothed by one of"):
        smoothed([], 6, how="smoothed-median")


def test_the_reading_refuses_a_refit_not_yet_whole_unless_partial(tmp_path: Path) -> None:
    conditions = {"arms": ["1x6", "6x6"], "spans": [6], "seeds": [1], "select_from": "2004-01"}
    (tmp_path / "conditions.json").write_text(json.dumps(conditions))
    (tmp_path / "chosen.json").write_text(
        json.dumps(
            [
                {
                    "family": STATES,
                    "detector": DIRECTION,
                    "arm": "6x6",
                    "best_context": "6x6",
                    "means": {},
                }
            ]
        )
    )
    write_cell(
        tmp_path,
        cell_name(REFIT, STATES, "1x6", DIRECTION, 1),
        REFIT,
        STATES,
        "1x6",
        DIRECTION,
        1,
        answers_scoring(False, 1),
    )

    with pytest.raises(SystemExit, match="refit cells are missing"):
        read_tables(tmp_path)
    missing = missing_refits(tmp_path, conditions)
    assert "refit-states-6x6-direction-s1" in missing
    assert "refit-smoothed-max-6h-direction-s1" in missing
    assert "Control" in read_tables(tmp_path, partial=True)


def test_the_choice_refuses_a_selection_not_yet_whole_unless_partial(tmp_path: Path) -> None:
    conditions = {"arms": ["1x6"], "spans": [6], "seeds": [1], "select_from": "2004-01"}
    (tmp_path / "conditions.json").write_text(json.dumps(conditions))
    write_cell(
        tmp_path,
        cell_name(SELECT, STATES, "1x6", DIRECTION, 1),
        SELECT,
        STATES,
        "1x6",
        DIRECTION,
        1,
        answers_scoring(False, 1),
    )

    with pytest.raises(SystemExit, match="selection cells are missing"):
        choose_from(tmp_path)
    assert [c.arm for c in choose_from(tmp_path, partial=True)] == ["1x6"]


def test_a_row_reads_the_arm_against_the_control_seeds_paired() -> None:
    control = {s: AnomalyRanking.of(answers_scoring(False, s)) for s in (1, 2, 3)}
    candidate = {s: AnomalyRanking.of(answers_scoring(True, s)) for s in (1, 2, 3)}

    row = context_row(STATES, DIRECTION, "6x6", candidate, control, chosen=True, bootstrap=QUICK)

    assert row.difference > 0.3
    assert row.low <= row.difference <= row.high
    assert row.outcome == "clearly above"
    assert set(row.areas) == {1, 2, 3}
    beside = context_row(
        STATES, DIRECTION, "24x6", candidate, control, chosen=False, bootstrap=QUICK
    )
    assert beside.outcome == "beside"
    judged = corrected([row, beside])
    assert judged[0].rejected is True
    assert judged[1].rejected is None
    with pytest.raises(SystemExit, match="lacks a seed"):
        context_row(
            STATES, DIRECTION, "6x6", candidate, {1: control[1]}, chosen=True, bootstrap=QUICK
        )


def test_the_events_found_are_those_with_a_window_among_the_top_normal_scores() -> None:
    answers = [predicted("m/1", p, float(p), 0.0, float(p)) for p in range(100)]
    answers += [predicted("m/1", 100, 100.0, 1.0, 98.5), predicted("m/1", 101, 101.0, 2.0, 50.0)]

    assert events_found(answers, 0.01) == frozenset({1})
    assert events_found(answers, 0.0) == frozenset()
    assert events_found(answers, 0.5) == frozenset({1, 2})
    assert events_found([predicted("m/1", 0, 0.0, 1.0, 1.0)], 0.01) == frozenset()
    ranking = AnomalyRanking.of(answers)
    assert len(events_found(answers, 0.01)) / 2 == ranking.event_recall_at_false_alarm_share(0.01)
    tied = [predicted("m/1", p, float(p), 0.0, float(p // 10)) for p in range(100)]
    tied += [predicted("m/1", 100, 100.0, 1.0, 9.0), predicted("m/1", 101, 101.0, 2.0, 8.5)]
    for share in (0.01, 0.07, 0.1, 0.11, 0.29):
        found = events_found(tied, share)
        assert len(found) / 2 == AnomalyRanking.of(tied).event_recall_at_false_alarm_share(share)


def test_a_run_refuses_other_conditions_than_it_began_under(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    refuse_other_conditions(path, {"arms": ["1x6"], "commit": "a"})
    path.write_text(json.dumps({"arms": ["1x6"], "spans": [6], "commit": "a"}))
    refuse_other_conditions(path, {"arms": ["1x6"], "spans": [6], "commit": "b"})
    with pytest.raises(SystemExit, match="other conditions: spans"):
        refuse_other_conditions(path, {"arms": ["1x6"], "spans": [24], "commit": "a"})


def test_the_check_reports_distances_and_refuses_other_windows(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    own = answers_scoring(False, 1)
    write_answers(tmp_path / "answers" / "cell.csv", own)
    other = [
        predicted(
            str(p.window.unit), p.window.position, p.window.ends_at, p.target, p.predicted + 1e-6
        )
        for p in own
    ]
    write_answers(tmp_path / "other.csv", other)

    text = check_cell(tmp_path, "cell", tmp_path / "other.csv")

    assert "20 windows, targets agree" in text
    assert "within 1e-4: 1.0000" in text
    write_answers(tmp_path / "fewer.csv", other[:-1])
    with pytest.raises(SystemExit, match="1 differ"):
        check_cell(tmp_path, "cell", tmp_path / "fewer.csv")


def test_windows_tied_on_both_sides_move_no_rank_whatever_their_level(tmp_path: Path) -> None:
    (tmp_path / "answers").mkdir()
    own = answers_scoring(False, 1)
    tied = [
        predicted(str(p.window.unit), p.window.position, p.window.ends_at, p.target, score)
        for p, score in zip(own, [0.0] * 14 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True)
    ]
    write_answers(tmp_path / "answers" / "cell.csv", tied)
    shifted = [
        predicted(
            str(p.window.unit), p.window.position, p.window.ends_at, p.target, p.predicted - 0.37
        )
        for p in tied
    ]
    write_answers(tmp_path / "other.csv", shifted)
    assert "ranks moved by more than 10: 0" in check_cell(tmp_path, "cell", tmp_path / "other.csv")


def test_main_routes_the_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    conditions = {"arms": ["1x6"], "spans": [6], "seeds": [1], "select_from": "2004-01"}
    (tmp_path / "conditions.json").write_text(json.dumps(conditions))
    write_cell(
        tmp_path,
        cell_name(SELECT, STATES, "1x6", DIRECTION, 1),
        SELECT,
        STATES,
        "1x6",
        DIRECTION,
        1,
        answers_scoring(False, 1),
    )

    main(["choose", "--dir", str(tmp_path), "--partial"])

    assert "Chosen" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["measure", "--out", str(tmp_path)])


def test_a_window_whose_context_reaches_an_unloaded_month_is_not_read(tmp_path: Path) -> None:
    origin = 946_684_800  # 2000-01-01T00:00Z
    assert month_of_hour(1, origin) == "2000-01"
    assert month_of_hour(31 * 24, origin) == "2000-01"
    assert month_of_hour(31 * 24 + 1, origin) == "2000-02"
    march = 31 * 24 + 29 * 24  # 2000 is a leap year: March starts after 60 days
    windows = [
        labelled_at("m/2000-03", 0, float(march + 1), 0.0),
        labelled_at("m/2000-03", 1, float(march + 5), 1.0),
        labelled_at("m/2000-03", 2, float(march + 6), 0.0),
        labelled_at("m/2000-01", 3, 10.0, 0.0),
    ]

    truncated = truncated_windows(windows, {"2000-01", "2000-03"}, 6, origin)

    assert [w.window.position for w in truncated] == [0, 1]
    assert truncated_windows(windows, {"2000-01", "2000-02", "2000-03"}, 6, origin) == []
    assert truncated_windows(windows, {"2000-01", "2000-03"}, 1, origin) == []
    write_truncated(tmp_path / "truncated-refit.csv", truncated)
    dropped = read_truncated(tmp_path, "refit")
    assert dropped == {("m/2000-03", 0), ("m/2000-03", 1)}
    assert read_truncated(tmp_path, "select") == set()
    answers = [predicted("m/2000-03", p, 0.0, 0.0, 1.0) for p in range(3)]
    assert [p.window.position for p in kept(answers, dropped)] == [2]


def labelled_at(unit: str, position: int, ends_at: float, target: float) -> LabelledWindow:
    return LabelledWindow(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at), target=target
    )


def test_the_outcome_needs_holm_and_says_when_the_threshold_exceeds_the_least_gain() -> None:
    control = {s: AnomalyRanking.of(answers_scoring(False, s)) for s in (1, 2, 3)}
    candidate = {s: AnomalyRanking.of(answers_scoring(True, s)) for s in (1, 2, 3)}
    row = context_row(STATES, DIRECTION, "6x6", candidate, control, chosen=True, bootstrap=QUICK)

    assert row.outcome == "clearly above"
    assert replace(row, rejected=False).outcome == "above, not under Holm"
    undecided = replace(row, difference=0.01, low=-0.05, high=0.07, threshold=0.06)
    assert undecided.outcome == "undecided"
    assert replace(undecided, threshold=0.04).outcome == "around"
