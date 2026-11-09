"""The detection rule's power is simulated with the harness's own sweep, on a known structure."""

from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_event_recalls import PairedEventRecalls
from scripts.detection_power_report import (
    INJECTED_POWER,
    INTERACTION_POWER,
    JITTER_WIDTHS,
    JITTERED,
    ONSET,
    POWER,
    RECALL_POWER,
    SHARED,
    STRUCTURE,
    Structure,
    average_precision,
    check_against_the_harness,
    confirms,
    confirms_interaction,
    confirms_recall,
    draw_sides,
    event_recalls,
    found_events,
    injected_candidate,
    main,
    read_pairs,
    resampled_counts,
    signal_windows,
)


def write_structure(path: Path, units: int, windows: int, events: int) -> Path:
    """``units`` units of ``windows`` windows, the first ``events`` units holding one event each."""
    lines = ["unit,position,event"]
    position = 0
    for unit in range(units):
        for _ in range(windows):
            lines.append(f"u{unit:02d},{position},{unit + 1 if unit < events else 0}")
            position += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_sweep_agrees_with_the_harness() -> None:
    check_against_the_harness()


def test_an_event_weighs_one_however_many_windows_it_spans() -> None:
    structure = Structure(
        units=np.array([0, 0, 0, 1, 1]), events=np.array([1, 1, 1, 0, 2]), unit_count=2
    )

    assert structure.weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])
    assert [stratum.tolist() for stratum in structure.strata] == [[0, 1], []]
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # The three-window event is found whole before the normal window, the one-window event
    # after it: the mean of a precision of one and of two thirds, each event weighing one.
    area = average_precision(scores, structure.events, structure.weights, np.ones(5))
    assert area == pytest.approx((1.0 + 2 / 3) / 2)


def test_a_candidate_no_better_than_the_control_is_not_confirmed(tmp_path: Path) -> None:
    structure = Structure.read(write_structure(tmp_path / "s.csv", units=12, windows=4, events=4))
    draws = np.random.default_rng(1)
    control = draws.standard_normal(48)

    assert not confirms(structure, control, control, minimum=0.02, resamples=50, draws=draws)


def test_the_report_writes_the_power_per_level_and_gain(tmp_path: Path) -> None:
    structure = write_structure(tmp_path / "s.csv", units=12, windows=4, events=4)

    main(
        [
            "simulate",
            "--structure",
            str(structure),
            "--out",
            str(tmp_path),
            "--level",
            "0.3",
            "--gains",
            "0.1",
            "--worlds",
            "4",
            "--resamples",
            "20",
        ]
    )

    written = (tmp_path / POWER).read_text().splitlines()
    assert written[0] == "level,gain,minimum,worlds,power"
    assert written[1].startswith("0.3,0.1,0.02,4,")
    assert STRUCTURE == "structure.csv"


def test_an_equal_lift_on_both_sides_is_no_interaction_and_a_lift_on_one_side_is(
    tmp_path: Path,
) -> None:
    structure = Structure.read(write_structure(tmp_path / "s.csv", units=12, windows=4, events=4))
    draws = np.random.default_rng(2)
    control = draws.standard_normal(48)
    lifted = control + np.where(structure.events > 0, 1.0, 0.0)
    raised = control + np.where(structure.events > 0, 4.0, 0.0)

    assert not confirms_interaction(
        structure, (control, control, lifted, lifted), minimum=0.02, resamples=50, draws=draws
    )
    assert confirms_interaction(
        structure, (control, control, lifted, raised), minimum=0.02, resamples=50, draws=draws
    )


def test_the_interaction_report_writes_the_power_per_level_teaching_and_gain(
    tmp_path: Path,
) -> None:
    structure = write_structure(tmp_path / "s.csv", units=12, windows=4, events=4)

    main(
        [
            "interaction",
            "--structure",
            str(structure),
            "--out",
            str(tmp_path),
            "--level",
            "0.3",
            "--teaching",
            "0.1",
            "--gains",
            "0.2",
            "--worlds",
            "4",
            "--resamples",
            "20",
        ]
    )

    written = (tmp_path / INTERACTION_POWER).read_text().splitlines()
    assert written[0] == "level,teaching,gain,minimum,worlds,power"
    assert written[1].startswith("0.3,0.1,0.2,0.02,4,")


def ranking_of(structure: Structure, scores: np.ndarray) -> AnomalyRanking:
    return AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(
                unit=UnitKey(f"u{unit:03d}"), position=position, ends_at=float(position)
            ),
            target=float(event),
            predicted=float(score),
        )
        for position, (unit, event, score) in enumerate(
            zip(structure.units.tolist(), structure.events.tolist(), scores.tolist(), strict=True)
        )
    )


def small_structure(draws: np.random.Generator) -> Structure:
    """Twenty units; events of one to six windows, some spanning a unit's boundary."""
    units = np.repeat(np.arange(20), 30)
    events = np.zeros(len(units), dtype=np.int64)
    number, at = 1, 4
    while at < len(units) - 8:
        length = int(draws.integers(1, 7))
        events[at : at + length] = number
        number += 1
        at += length + int(draws.integers(20, 40))
    return Structure(units=units, events=events, unit_count=20)


@pytest.mark.parametrize("share", [0.0, 0.01, 0.1, 0.5])
def test_the_recall_agrees_with_the_harness_including_ties(share: float) -> None:
    draws = np.random.default_rng(3)
    structure = small_structure(draws)
    # Rounding to a coarse grid leaves many windows tied, across events and normal windows.
    scores = np.round(draws.standard_normal(len(structure.events)) * 2.0) / 2.0
    once = np.ones((1, structure.unit_count))
    assert event_recalls(structure, scores, once, share)[0] == pytest.approx(
        ranking_of(structure, scores).event_recall_at_false_alarm_share(share), abs=1e-12
    )


def test_the_recall_over_draws_of_units_agrees_with_the_harness() -> None:
    draws = np.random.default_rng(4)
    structure = small_structure(draws)
    control = draws.standard_normal(len(structure.events))
    candidate = control + (structure.events > 0) * 1.0
    paired = PairedEventRecalls(
        rankings=PairedAnomalyRankings(
            control=(ranking_of(structure, control),),
            candidate=(ranking_of(structure, candidate),),
        ),
        share=0.05,
    )
    counts = resampled_counts(structure, 50, draws)
    picks = [
        [unit for unit, count in enumerate(row.astype(int).tolist()) for _ in range(count)]
        for row in counts
    ]
    got = event_recalls(structure, candidate, counts, 0.05) - event_recalls(
        structure, control, counts, 0.05
    )
    assert got.tolist() == pytest.approx(paired.reductions_over(picks), abs=1e-12)


def test_a_draw_that_leaves_no_normal_window_is_refused() -> None:
    structure = Structure(units=np.array([0, 1]), events=np.array([1, 0]), unit_count=2)
    with pytest.raises(ValueError, match="no normal window"):
        event_recalls(structure, np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), 0.1)


def test_the_onset_model_shifts_only_each_events_first_window() -> None:
    structure = Structure(
        units=np.array([0, 0, 0, 1, 1]), events=np.array([1, 1, 0, 2, 2]), unit_count=2
    )
    assert signal_windows(structure, ONSET).tolist() == [True, False, False, True, False]
    assert signal_windows(structure, JITTERED).tolist() == [True, True, False, True, True]


def test_without_jitter_the_candidate_never_loses_an_event() -> None:
    draws = np.random.default_rng(5)
    structure = small_structure(draws)
    signal = signal_windows(structure, SHARED)
    once = np.ones((1, structure.unit_count))
    for _ in range(20):
        control, candidate = draw_sides(
            structure, signal, base=1.0, raised=2.0, jitter=0.0, draws=draws
        )
        assert (
            event_recalls(structure, candidate, once, 0.01)[0]
            >= event_recalls(structure, control, once, 0.01)[0]
        )


def test_jitter_can_lose_an_event() -> None:
    draws = np.random.default_rng(6)
    structure = small_structure(draws)
    signal = signal_windows(structure, JITTERED)
    once = np.ones((1, structure.unit_count))
    gains = []
    for _ in range(50):
        control, candidate = draw_sides(
            structure, signal, base=2.5, raised=2.5, jitter=3.0, draws=draws
        )
        gains.append(
            event_recalls(structure, candidate, once, 0.05)[0]
            - event_recalls(structure, control, once, 0.05)[0]
        )
    assert min(gains) < 0.0 < max(gains)


@pytest.mark.parametrize(
    ("gain", "lower", "expected"),
    [(0.05, 0.01, True), (0.01, 0.01, False), (0.05, -0.01, False), (0.016, 0.0, False)],
)
def test_the_recall_rule_needs_the_floor_and_an_interval_above_zero(
    gain: float, lower: float, expected: bool
) -> None:
    assert confirms_recall(gain, lower, minimum=0.016) is expected


def test_the_recall_report_writes_power_per_model(tmp_path: Path) -> None:
    path = write_structure(tmp_path / STRUCTURE, 10, 40, 5)
    main(
        [
            "recall",
            "--structure",
            str(path),
            "--out",
            str(tmp_path),
            "--share",
            "0.05",
            "--level",
            "0.5",
            "--gains",
            "0.2",
            "--models",
            SHARED,
            ONSET,
            "--jitter",
            "1",
            "--worlds",
            "2",
            "--resamples",
            "20",
        ]
    )
    rows = (tmp_path / RECALL_POWER).read_text().splitlines()
    assert rows[0] == "model,jitter,level,gain,minimum,worlds,power,half_width,realised_gain"
    assert [row.split(",")[0] for row in rows[1:]] == [SHARED, ONSET]


def test_the_jitter_report_writes_a_width_per_model_and_jitter(tmp_path: Path) -> None:
    path = write_structure(tmp_path / STRUCTURE, 10, 40, 5)
    main(
        [
            "jitter",
            "--structure",
            str(path),
            "--out",
            str(tmp_path),
            "--share",
            "0.05",
            "--level",
            "0.5",
            "--jitters",
            "0",
            "1",
            "--worlds",
            "2",
            "--resamples",
            "20",
        ]
    )
    rows = (tmp_path / JITTER_WIDTHS).read_text().splitlines()
    assert len(rows) == 5


def separated_pair() -> tuple[Structure, np.ndarray, np.ndarray]:
    """Ten events of two windows in ten units; both detectors find the first four only."""
    units = np.repeat(np.arange(10), 100)
    events = np.zeros(1000, dtype=np.int64)
    for unit in range(10):
        events[unit * 100 + 10 : unit * 100 + 12] = unit + 1
    draws = np.random.default_rng(1)
    control = draws.uniform(0.0, 1.0, size=1000)
    control[events > 0] = 0.1
    control[np.isin(events, [1, 2, 3, 4])] = 5.0
    return Structure(units=units, events=events, unit_count=10), control, control.copy()


def test_the_injection_reaches_the_gain_asked_for_and_touches_only_missed_events() -> None:
    structure, control, candidate = separated_pair()
    raised = injected_candidate(
        structure, control, candidate, gain=0.3, share=0.01, draws=np.random.default_rng(2)
    )
    assert len(found_events(structure, raised, 0.01)) - 4 == 3
    changed = np.flatnonzero(raised != candidate)
    assert len(changed) == 3
    assert not set(structure.events[changed]) & {0, 1, 2, 3, 4}


def test_the_injection_counts_a_gain_the_pair_already_has() -> None:
    structure, control, candidate = separated_pair()
    candidate[structure.events == 5] = 5.0
    raised = injected_candidate(
        structure, control, candidate, gain=0.3, share=0.01, draws=np.random.default_rng(2)
    )
    assert len(found_events(structure, raised, 0.01)) == 7


def test_no_injection_is_made_when_the_gain_is_already_there() -> None:
    structure, control, candidate = separated_pair()
    candidate[structure.events == 5] = 5.0
    raised = injected_candidate(
        structure, control, candidate, gain=0.1, share=0.01, draws=np.random.default_rng(2)
    )
    assert np.array_equal(raised, candidate)


def test_a_gain_beyond_the_missed_events_is_refused() -> None:
    structure, control, candidate = separated_pair()
    with pytest.raises(ValueError, match="cannot find"):
        injected_candidate(
            structure, control, candidate, gain=0.9, share=0.01, draws=np.random.default_rng(2)
        )


def write_predictions(path: Path, *, drop_last: bool = False) -> None:
    lines = ["campaign,candidate,budget,seed,unit,position,ends_at,target,predicted"]
    windows = [("u0", 0, 0), ("u0", 1, 1), ("u1", 2, 0), ("u1", 3, 2)]
    for campaign, candidate in (("c1", "a"), ("c2", "b,x")):
        for seed in (1, 2):
            for unit, position, target in windows:
                lines.append(
                    f'{campaign},"{candidate}",all,{seed},{unit},{position},0.0,{target}.0,'
                    f"{0.1 * position + seed}"
                )
    if drop_last:
        lines = [line for line in lines if not line.startswith('c2,"b,x",all,1,u1,3')]
    path.write_text("\n".join(lines) + "\n")


def test_the_pairs_are_aligned_and_read_at_the_first_seed(tmp_path: Path) -> None:
    predictions, plan = tmp_path / "p.csv", tmp_path / "plan.tsv"
    write_predictions(predictions)
    plan.write_text("c1\ta\tc2\tb,x\n")
    structure, pairs = read_pairs(predictions, plan)
    assert structure.events.tolist() == [0, 1, 0, 2]
    assert structure.units.tolist() == [0, 0, 1, 1]
    assert pairs[0][1].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3])


def test_a_detector_short_of_a_window_is_refused(tmp_path: Path) -> None:
    predictions, plan = tmp_path / "p.csv", tmp_path / "plan.tsv"
    write_predictions(predictions, drop_last=True)
    plan.write_text("c1\ta\tc2\tb,x\n")
    with pytest.raises(SystemExit, match="scores"):
        read_pairs(predictions, plan)


def test_the_injected_report_writes_a_row_per_gain(tmp_path: Path) -> None:
    structure, control, candidate = separated_pair()
    lines = ["campaign,candidate,budget,seed,unit,position,ends_at,target,predicted"]
    for name, scores in (("c1", control), ("c2", candidate)):
        for at, (unit, event, score) in enumerate(
            zip(structure.units.tolist(), structure.events.tolist(), scores.tolist(), strict=True)
        ):
            lines.append(f"{name},d,all,1,u{unit},{at},0.0,{event}.0,{score}")
    predictions, plan = tmp_path / "p.csv", tmp_path / "plan.tsv"
    predictions.write_text("\n".join(lines) + "\n")
    plan.write_text("c1\td\tc2\td\n")
    main(
        [
            "injected",
            "--predictions",
            str(predictions),
            "--pairs",
            str(plan),
            "--out",
            str(tmp_path),
            "--gains",
            "0.2",
            "0.3",
            "--worlds",
            "3",
            "--resamples",
            "20",
        ]
    )
    assert len((tmp_path / INJECTED_POWER).read_text().splitlines()) == 3
