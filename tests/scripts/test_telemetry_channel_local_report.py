"""The channel-local report: departures, their scores, the choice and the reading, on CPU."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.taught_fine_tuning_pilot import write_answers
from scripts.telemetry_channel_local_report import (
    CELL_COLUMNS,
    CELLS,
    COMBINED,
    DECLARED,
    FLOOR_SHARE,
    LARGEST,
    LOCAL,
    MAX,
    MEAN,
    NO_SEED,
    STATES,
    TAUGHT,
    Arm,
    Departures,
    after_event_share,
    agreement_of,
    arm_cells,
    arms_of,
    bin_values,
    cell_name,
    choose,
    choose_from,
    chosen_of,
    control_cells,
    corrected,
    event_annotations,
    global_floor,
    hour_summaries,
    largest_departures,
    local_deviations,
    main,
    missing_refits,
    missing_selects,
    paired,
    read_tables,
    reading_row,
    refit_arms,
    window_scores,
)
from scripts.telemetry_context_report import (
    BASE_BINS,
    DIRECTION,
    FARTHEST,
    REFIT,
    SELECT,
    TREES,
    WHITENED,
    PublishedCells,
)

QUICK = PairedUnitBootstrap(resamples=400, seed=1)
F = BinnedTokens.N_FEATURES
PRESENCE, LOG_COUNT = BinnedTokens.PRESENCE, BinnedTokens.LOG_COUNT
MEAN_AT, MIN_AT, MAX_AT, LAST_AT = (
    BinnedTokens.FEATURE_NAMES.index(name) for name in ("mean", "minimum", "maximum", "last")
)


def cell(bins: list[tuple[float, float, float, float, int] | None]) -> np.ndarray:
    """One channel's cells over the bins: (mean, min, max, last, readings) or an empty bin."""
    out = np.zeros((len(bins), F))
    for place, each in enumerate(bins):
        if each is None:
            continue
        mean, least, most, last, readings = each
        out[place, PRESENCE] = 1.0
        out[place, LOG_COUNT] = np.log1p(readings)
        out[place, MEAN_AT], out[place, MIN_AT], out[place, MAX_AT], out[place, LAST_AT] = (
            mean,
            least,
            most,
            last,
        )
    return out


def test_an_arm_is_named_by_its_features_and_span_and_needs_half_its_context() -> None:
    arm = Arm.parse("hour-24h")

    assert arm == Arm("hour", 24)
    assert arm.name == "hour-24h"
    assert arm.least_context == 12
    assert Arm.parse("bins-6h").least_context == 3
    assert [a.name for a in arms_of(["hour", "bins"], [6, 24])] == [
        "hour-6h",
        "hour-24h",
        "bins-6h",
        "bins-24h",
    ]


@pytest.mark.parametrize("text", ["hour", "day-24h", "hour-24", "hour-1h", "hour-xh"])
def test_an_arm_not_of_the_form_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match=r"arm|span"):
        Arm.parse(text)


def test_an_hour_is_summarised_over_its_present_bins_and_missing_where_none_is() -> None:
    cells = np.stack(
        [
            cell([(1.0, 0.0, 2.0, 1.5, 3), None, (3.0, 2.0, 4.0, 3.5, 1)]),
            cell([None, None, None]),
        ]
    )[None]

    summaries = hour_summaries(cells)

    assert summaries.shape == (1, 2, 5)
    np.testing.assert_allclose(summaries[0, 0], [2.0, 0.0, 4.0, 3.5, np.log1p(4.0)])
    assert np.isnan(summaries[0, 1]).all()


def test_the_bins_values_are_missing_where_the_bin_holds_nothing() -> None:
    cells = cell([(1.0, 0.0, 2.0, 1.5, 3), None])[None, None]

    values = bin_values(cells)

    assert values.shape == (1, 1, 2, 5)
    np.testing.assert_allclose(values[0, 0, 0], [1.0, 0.0, 2.0, 1.5, np.log1p(3.0)])
    assert np.isnan(values[0, 0, 1]).all()


def test_the_floor_is_a_share_of_each_channels_spread_or_of_the_typical_one() -> None:
    summaries = np.full((5, 2, 5), np.nan)
    summaries[:, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]
    summaries[:, 1, 0] = 7.0
    summaries[:, 0, 1] = [0.0, 10.0, 20.0, 30.0, 40.0]

    floor = global_floor(summaries)

    mad_a, mad_b = 1.4826 * 1.0, 1.4826 * 10.0
    typical = np.median([mad_a, mad_b])
    assert floor[0, 0] == pytest.approx(FLOOR_SHARE * mad_a)
    assert floor[0, 1] == pytest.approx(FLOOR_SHARE * mad_b)
    assert floor[1, 0] == pytest.approx(FLOOR_SHARE * typical)
    assert floor[1, 1] == pytest.approx(FLOOR_SHARE * typical)
    assert global_floor(np.full((3, 1, 5), 2.0)).tolist() == [[1.0] * 5]


def test_a_departure_is_the_distance_from_the_recent_median_over_the_floored_spread() -> None:
    # Channel 0: context at 10 with spread, the hour at 13; channel 1: context constant, the
    # hour one above, read over the floor; channel 2: absent from the hour; channel 3: only one
    # context hour loaded.
    context = np.full((1, 4, 4, 5), np.nan)
    context[0, :, 0, 0] = [9.0, 10.0, 11.0, 10.0]
    context[0, :, 1, 0] = 5.0
    context[0, :, 2, 0] = 1.0
    context[0, 0, 3, 0] = 1.0
    now = np.full((1, 4, 1, 5), np.nan)
    now[0, 0, 0, 0] = 13.0
    now[0, 1, 0, 0] = 6.0
    now[0, 3, 0, 0] = 100.0
    floor = np.full((4, 5), 0.5)

    departures = local_deviations(now, context, floor, least_context=2)

    spread = max(1.4826 * 0.5, 0.5)
    assert departures.shape == (1, 4)
    assert departures[0, 0] == pytest.approx(3.0 / spread)
    assert departures[0, 1] == pytest.approx(1.0 / 0.5)
    assert departures[0, 2] == 0.0
    assert departures[0, 3] == 0.0


def test_a_windows_score_is_the_largest_departure_or_the_mean_over_the_present() -> None:
    departures = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    present = np.array([[True, True, False], [False, False, False]])

    assert window_scores(departures, present, MAX).tolist() == [3.0, 0.0]
    assert window_scores(departures, present, MEAN).tolist() == [2.0, 0.0]
    with pytest.raises(ValueError, match="not 'median'"):
        window_scores(departures, present, "median")


def test_the_largest_departures_are_the_top_few_in_order_logged() -> None:
    departures = np.array([[0.0, 5.0, 2.0, 9.0], [1.0, 0.0, 0.0, 0.0]])

    largest = largest_departures(departures, 2)

    np.testing.assert_allclose(largest, np.log1p([[9.0, 5.0], [1.0, 0.0]]))
    assert largest_departures(departures).shape == (2, LARGEST)


def hourly_cells(values: list[float | None], readings: int = 2) -> np.ndarray:
    """One channel, one window per hour, every bin at the hour's value, or an empty hour."""
    out = np.zeros((len(values), 1, BASE_BINS, F))
    for hour, value in enumerate(values):
        if value is None:
            continue
        out[hour, 0] = cell([(value, value - 0.5, value + 0.5, value, readings)] * BASE_BINS)
    return out


def test_the_departures_of_every_window_are_computed_in_chunks_and_kept(tmp_path: Path) -> None:
    values: list[float | None] = [1.0, 1.0, 1.0, 1.0, 1.0, None, 1.0, 4.0]
    published = PublishedCells(hourly_cells(values), np.array([3]), [float(h) for h in range(8)])
    departures = Departures(published, tmp_path / "departures")
    departures.CHUNK = 3
    fit = list(range(5))

    hour = departures.of(Arm.parse("hour-4h"), SELECT, fit)
    bins = departures.of(Arm.parse("bins-4h"), SELECT, fit)

    assert hour.shape == (8, 1)
    # The first windows lack half their context; the empty hour departs nought; the last hour
    # departs three over the floor, since its context never varied.
    assert hour[:2, 0].tolist() == [0.0, 0.0]
    assert hour[5, 0] == 0.0
    assert hour[7, 0] > hour[6, 0] == 0.0
    assert bins[7, 0] == pytest.approx(hour[7, 0])
    # The bins' context counts hours loaded, not bins: the second window has one of three.
    assert bins[:2, 0].tolist() == [0.0, 0.0]
    # The empty hour 5 is absent from the context of the windows at 6 and 7.
    assert departures.statistics(Arm.parse("bins-4h"))[2][:, 0].tolist() == [0, 1, 2, 3, 3, 3, 2, 2]
    assert (tmp_path / "departures" / "select-hour-4h.npy").exists()
    assert (tmp_path / "departures" / "bins-4h-centre.npy").exists()
    # The refit shares the context's statistics and applies its own floor.
    assert departures.of(Arm.parse("hour-4h"), REFIT, list(range(8))).shape == (8, 1)
    np.testing.assert_array_equal(np.load(tmp_path / "departures" / "select-bins-4h.npy"), bins)
    assert departures.present().tolist() == [[v is not None] for v in values]
    np.testing.assert_array_equal(departures.of(Arm.parse("hour-4h"), SELECT, fit), hour)


def test_the_choice_takes_the_shorter_span_then_the_hour_within_the_margin() -> None:
    choice = choose(LOCAL, MAX, {"hour-24h": 0.30, "bins-6h": 0.295, "hour-6h": 0.292})

    assert choice.arm == "hour-6h"
    assert choose(LOCAL, MAX, {"hour-24h": 0.30, "bins-6h": 0.25}).arm == "hour-24h"
    with pytest.raises(ValueError, match="no arm"):
        choose(LOCAL, MAX, {})


def test_the_cells_of_an_arm_are_seedless_where_the_detector_draws_nothing() -> None:
    cells = arm_cells(SELECT, [Arm.parse("hour-6h")], [1, 2])

    assert (LOCAL, MAX, "hour-6h", NO_SEED) in cells
    assert (TAUGHT, DIRECTION, "hour-6h", NO_SEED) in cells
    assert {s for f, d, a, s in cells if d == COMBINED} == {1, 2}
    assert {s for f, d, a, s in cells if d == TREES} == {1, 2}
    assert (TAUGHT, "top", "hour-6h", NO_SEED) in cells
    assert {s for f, d, a, s in cells if d == "ordered"} == {1, 2}
    assert len(cells) == 2 + 1 + 2 + 2 + 1 + 2
    assert control_cells([1, 2]) == [
        (STATES, DIRECTION, 1),
        (STATES, DIRECTION, 2),
        (STATES, WHITENED, 1),
        (STATES, WHITENED, 2),
        (CELLS, FARTHEST, NO_SEED),
    ]
    assert cell_name(REFIT, LOCAL, MAX, "hour-6h", NO_SEED) == "refit-local-max-hour-6h"
    assert cell_name(REFIT, TAUGHT, TREES, "hour-6h", 2) == "refit-taught-trees-hour-6h-s2"
    assert cell_name(REFIT, CELLS, FARTHEST, None, NO_SEED) == "refit-cells-farthest"


def predicted(
    unit: str, position: int, ends_at: float, target: float, score: float
) -> WindowPrediction:
    return WindowPrediction(
        window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at),
        target=target,
        predicted=score,
    )


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


def test_a_seedless_side_stands_beside_every_seed_of_the_other() -> None:
    one = AnomalyRanking.of(answers_scoring(True, 1))
    two = AnomalyRanking.of(answers_scoring(False, 2))
    three = AnomalyRanking.of(answers_scoring(False, 3))

    candidate, control = paired({NO_SEED: one}, {1: two, 2: three})
    assert candidate == {1: one, 2: one}
    assert control == {1: two, 2: three}
    candidate, control = paired({1: two, 2: three}, {NO_SEED: one})
    assert candidate == {1: two, 2: three}
    assert control == {1: one, 2: one}
    candidate, control = paired({NO_SEED: one}, {NO_SEED: two})
    assert candidate == {NO_SEED: one}
    assert control == {NO_SEED: two}


def test_a_row_of_two_seedless_sides_has_no_threshold_and_one_pair() -> None:
    candidate = {NO_SEED: AnomalyRanking.of(answers_scoring(True, 1))}
    control = {NO_SEED: AnomalyRanking.of(answers_scoring(False, 1))}

    row = reading_row(LOCAL, MAX, "hour-6h", candidate, control, chosen=True, bootstrap=QUICK)

    assert row.threshold == 0.0
    assert list(row.areas) == [NO_SEED]
    assert row.difference > 0.0
    assert row.low <= row.difference <= row.high


def test_a_seedless_arm_against_a_seeded_control_takes_the_controls_spread() -> None:
    candidate = {NO_SEED: AnomalyRanking.of(answers_scoring(True, 1))}
    control = {s: AnomalyRanking.of(answers_scoring(False, s)) for s in (1, 2, 3)}

    row = reading_row(
        TAUGHT, DIRECTION, "hour-6h", candidate, control, chosen=True, bootstrap=QUICK
    )

    assert sorted(row.areas) == [1, 2, 3]
    assert len(set(row.areas.values())) == 1
    assert row.threshold > 0.0
    with pytest.raises(SystemExit, match="lacks a seed"):
        reading_row(
            TAUGHT,
            COMBINED,
            "hour-6h",
            {1: control[1], 4: control[2]},
            control,
            chosen=True,
            bootstrap=QUICK,
        )


def test_the_declared_rows_are_judged_as_one_family_under_holm() -> None:
    good = {NO_SEED: AnomalyRanking.of(answers_scoring(True, 1))}
    bad = {NO_SEED: AnomalyRanking.of(answers_scoring(False, 1))}
    rows = [
        reading_row(LOCAL, MAX, "hour-6h", good, bad, chosen=True, bootstrap=QUICK),
        reading_row(TAUGHT, COMBINED, "hour-6h", good, bad, chosen=True, bootstrap=QUICK),
        reading_row(TAUGHT, DIRECTION, "hour-6h", good, bad, chosen=True, bootstrap=QUICK),
        reading_row(LOCAL, MAX, "bins-6h", good, bad, chosen=False, bootstrap=QUICK),
    ]

    judged = corrected(rows)

    assert [row.rejected for row in judged] == [True, True, None, None]
    assert {(row.family, row.detector) for row in judged if row.rejected is not None} == set(
        DECLARED
    )
    assert judged[0].outcome == "clearly above"
    assert judged[3].outcome == "beside"


def write_cell(
    directory: Path,
    section: str,
    family: str,
    detector: str,
    arm: str | None,
    seed: int,
    answers: list[WindowPrediction],
) -> str:
    name = cell_name(section, family, detector, arm, seed)
    table_path = directory / "cells.csv"
    new = not table_path.exists()
    with table_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(CELL_COLUMNS)
        writer.writerow(
            (name, section, family, detector, arm or "", seed, 3, 10, len(answers), "1.0")
        )
    (directory / "answers").mkdir(exist_ok=True)
    write_answers(directory / "answers" / f"{name}.csv", answers)
    return name


CONDITIONS = {
    "features": ["hour", "bins"],
    "spans": [6],
    "seeds": [1, 2],
    "select_from": "2004-01",
    "widest_hours": 6,
    "windows_read": 20,
}


def a_section(directory: Path, section: str, arms: list[str], *, good: str) -> None:
    for arm in arms:
        for family, detector, name, seed in arm_cells(section, [Arm.parse(arm)], [1, 2]):
            write_cell(
                directory,
                section,
                family,
                detector,
                name,
                seed,
                answers_scoring(arm == good, seed if seed != NO_SEED else 1),
            )
    for family, detector, seed in control_cells([1, 2]):
        write_cell(
            directory,
            section,
            family,
            detector,
            None,
            seed,
            answers_scoring(False, seed if seed != NO_SEED else 1),
        )


def test_the_choice_from_the_selection_and_the_reading_on_the_refit(tmp_path: Path) -> None:
    (tmp_path / "conditions.json").write_text(json.dumps(CONDITIONS))
    for section in (SELECT, REFIT):
        (tmp_path / f"truncated-{section}.csv").write_text("unit,position,ends_at,target\n")
    a_section(tmp_path, SELECT, ["hour-6h", "bins-6h"], good="bins-6h")
    assert missing_selects(tmp_path, CONDITIONS) == []

    choices = choose_from(tmp_path)

    by_key = {(c.family, c.detector): c for c in choices}
    assert set(by_key) == {(LOCAL, MAX), (LOCAL, MEAN)} | {
        (TAUGHT, d) for d in (DIRECTION, COMBINED, TREES, "top", "ordered")
    }
    assert all(c.arm == "bins-6h" for c in choices)
    (tmp_path / "chosen.json").write_text(json.dumps([c.__dict__ for c in choices]))
    assert [a.name for a in refit_arms(chosen_of(tmp_path))] == ["bins-6h"]
    assert missing_refits(tmp_path, CONDITIONS) != []
    a_section(tmp_path, REFIT, ["bins-6h"], good="bins-6h")
    assert missing_refits(tmp_path, CONDITIONS) == []

    text = read_tables(tmp_path, {1: "Local", 2: "Global", 3: "Local", 4: "Local"})

    assert (tmp_path / "reading.md").exists()
    declared = [line for line in text.splitlines() if line.startswith("| local | max | bins-6h |")]
    assert len(declared) == 1
    assert "clearly above" in declared[0]
    assert "passes" in declared[0]
    assert "| taught | combined | bins-6h |" in text
    assert "| taught | trees | bins-6h |" in text
    assert "| states | direction |" in text
    assert "| cells | farthest |" in text
    assert "under every arm alike" in text
    assert "| refit-local-max-bins-6h | 4 | 3 | 0 |" in text


def test_the_reading_and_the_refit_refuse_a_run_not_yet_whole(tmp_path: Path) -> None:
    (tmp_path / "conditions.json").write_text(json.dumps(CONDITIONS))
    with pytest.raises(SystemExit, match="choose first"):
        chosen_of(tmp_path)
    with pytest.raises(SystemExit, match="selection cells are missing"):
        choose_from(tmp_path)
    (tmp_path / "chosen.json").write_text(
        json.dumps([{"family": LOCAL, "detector": MAX, "arm": "hour-6h", "means": {}}])
    )
    with pytest.raises(SystemExit, match="refit cells are missing"):
        read_tables(tmp_path, {})


def test_the_events_are_numbered_as_the_ground_truth_numbers_them(tmp_path: Path) -> None:
    mission = tmp_path / "M"
    mission.mkdir()
    (mission / "labels.csv").write_text(
        "ID,Channel,StartTime,EndTime\n"
        "id_1,channel_1,2000-01-01T00:00:00Z,2000-01-01T01:00:00Z\n"
        "id_2,channel_9,2000-01-02T00:00:00Z,2000-01-02T01:00:00Z\n"
        "id_3,channel_2,2000-01-03T00:00:00Z,2000-01-03T01:00:00Z\n"
        "id_3,channel_1,2000-01-03T00:00:00Z,2000-01-03T01:00:00Z\n"
        "id_4,channel_1,2000-01-04T00:00:00Z,2000-01-04T01:00:00Z\n"
        "id_5,telecommand_1,2000-01-05T00:00:00Z,2000-01-05T01:00:00Z\n"
    )
    (mission / "anomaly_types.csv").write_text(
        "ID,Class,Subclass,Category,Dimensionality,Locality,Length\n"
        "id_1,class_1,subclass_1,Anomaly,Multivariate,Local,Subsequence\n"
        "id_2,class_1,subclass_1,Anomaly,Multivariate,Global,Subsequence\n"
        "id_3,class_2,subclass_1,Communication Gap,Multivariate,Global,Subsequence\n"
        "id_4,class_3,subclass_1,Rare Event,Multivariate,Global,Point\n"
        "id_5,class_3,subclass_1,Anomaly,Multivariate,Local,Point\n"
    )

    # A row naming no channel is refused, as the ground truth's reader refuses it; without it,
    # channel 9 is not read so its event is left out, and the gap is not numbered.
    with pytest.raises(ValueError, match="names no channel"):
        event_annotations(tmp_path, "M", [1, 2])
    lines = (mission / "labels.csv").read_text().splitlines()
    (mission / "labels.csv").write_text("\n".join(lines[:-1]) + "\n")
    assert event_annotations(tmp_path, "M", [1, 2]) == {
        1: ("class_1", "Local"),
        2: ("class_3", "Global"),
    }


def test_the_share_of_top_false_alarms_after_an_event_counts_the_hours_since_its_last_window() -> (
    None
):
    # An event at hours 10 and 11; normal windows at 12, 13, 20 scored highest, 5 and 30 low.
    answers = [
        predicted("m", 0, 5.0, 0.0, 0.1),
        predicted("m", 1, 10.0, 1.0, 9.0),
        predicted("m", 2, 11.0, 1.0, 9.0),
        predicted("m", 3, 12.0, 0.0, 5.0),
        predicted("m", 4, 13.0, 0.0, 4.0),
        predicted("m", 5, 20.0, 0.0, 3.0),
        predicted("m", 6, 30.0, 0.0, 0.2),
    ]

    assert after_event_share(answers, 6, 0.6) == pytest.approx(2 / 3)
    assert after_event_share(answers, 12, 0.6) == pytest.approx(1.0)
    assert after_event_share(answers, 1, 0.6) == pytest.approx(1 / 3)
    assert after_event_share([predicted("m", 0, 5.0, 0.0, 0.1)], 6, 0.5) == 0.0


def test_main_routes_the_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "conditions.json").write_text(json.dumps({**CONDITIONS, "features": ["hour"]}))
    (tmp_path / "truncated-select.csv").write_text("unit,position,ends_at,target\n")
    a_section(tmp_path, SELECT, ["hour-6h"], good="hour-6h")

    main(["choose", "--dir", str(tmp_path)])

    assert (tmp_path / "chosen.json").exists()
    assert "| Family | Detector | hour-6h | Chosen |" in capsys.readouterr().out
    cell = cell_name(SELECT, LOCAL, MAX, "hour-6h", NO_SEED)
    main(
        [
            "check",
            "--dir",
            str(tmp_path),
            "--cell",
            cell,
            "--against",
            str(tmp_path / "answers" / f"{cell}.csv"),
        ]
    )
    out = capsys.readouterr().out
    assert "largest gap 0.00e+00" in out
    assert "Spearman 1.00000, top 1% held 1.000" in out
    other = tmp_path / "answers" / "reversed.csv"
    write_answers(
        other,
        [
            WindowPrediction(window=p.window, target=p.target, predicted=-p.predicted)
            for p in answers_scoring(True, 1)
        ],
    )
    with pytest.raises(SystemExit, match="less alike"):
        main(["check", "--dir", str(tmp_path), "--cell", cell, "--against", str(other)])
    # Shifted by a month-wise offset, the windows rank the same within every unit alone.
    shifted = tmp_path / "answers" / "shifted.csv"
    write_answers(
        shifted,
        [
            WindowPrediction(
                window=p.window,
                target=p.target,
                predicted=p.predicted + 1000.0 * (p.window.position // 5),
            )
            for p in answers_scoring(True, 1)
        ],
    )
    with pytest.raises(SystemExit, match="less alike"):
        main(["check", "--dir", str(tmp_path), "--cell", cell, "--against", str(shifted)])
    main(["check", "--dir", str(tmp_path), "--cell", cell, "--against", str(shifted), "--by-unit"])
    assert "least over 4 units: Spearman 1.00000, top 1% held 1.000" in capsys.readouterr().out


def test_tied_windows_share_their_mean_rank_as_spearmans_coefficient_defines_it() -> None:
    # A scoring that ties three windows against one that orders them: no whole agreement,
    # whatever order a sort leaves the tie in; Spearman's coefficient with mean ranks.
    tied, ordered = np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0])

    spearman, _ = agreement_of(tied, ordered)

    assert spearman == pytest.approx(np.corrcoef([2.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])[0, 1])
    # Two scorings with the same order and ties agree whole, at any level.
    level = np.concatenate((np.full(400, -2.0), np.linspace(0.0, 5.0, 344)))
    assert agreement_of(level, level - 0.37) == (1.0, 1.0)
