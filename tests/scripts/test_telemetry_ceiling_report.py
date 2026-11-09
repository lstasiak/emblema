"""Events are filed by the first family that finds them, and chance is held to one rotation."""

from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.detection_power_report import Structure, found_events
from scripts.telemetry_ceiling_report import (
    DEPARTURES,
    NOTHING,
    OTHER,
    REFIT,
    SELECT,
    TAUGHT,
    Detector,
    Excess,
    Sources,
    answer_detector,
    candidate_family,
    categories,
    chance_of_signal,
    channel_alarms,
    correlation,
    detector_found,
    event_signal,
    excess_over_rotations,
    family_found,
    found_mask,
    fused,
    fused_over_repeats,
    month_referenced,
    month_slices,
    queues_finer_cells,
    reopens_perturbation_pretext,
    resolution_found,
    rotated_counts,
    rotated_targets,
    rotation_order,
)


def structure_of_events(events: list[int], units: list[int]) -> Structure:
    return Structure(
        units=np.array(units, dtype=np.int64),
        events=np.array(events, dtype=np.int64),
        unit_count=max(units) + 1,
    )


def month_world() -> tuple[Structure, np.ndarray]:
    """Eight months of 100 windows, eight events of two windows at the start of each month."""
    units = np.repeat(np.arange(8), 100)
    events = np.zeros(800, dtype=np.int64)
    for month in range(8):
        events[month * 100 + 40 : month * 100 + 42] = month + 1
    return Structure(units=units, events=events, unit_count=8), np.arange(1, 9, dtype=np.int64)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("refit-local-max-hour-6h", (REFIT, "refit-local-max-hour-6h", DEPARTURES, 1)),
        ("select-local-mean-bins-72h", (SELECT, "select-local-mean-bins-72h", DEPARTURES, 1)),
        ("refit-states-whitened-s2", (REFIT, "refit-states-whitened", OTHER, 2)),
        ("refit-cells-farthest", (REFIT, "refit-cells-farthest", OTHER, 1)),
        ("refit-states-direction-s3", (REFIT, "refit-states-direction", TAUGHT, 3)),
        ("refit-taught-trees-hour-6h-s1", (REFIT, "refit-taught-trees-hour-6h", TAUGHT, 1)),
        ("refit-cells-1x6-trees-s2", (REFIT, "refit-cells-1x6-trees", TAUGHT, 2)),
        ("direction-base1-k99-s1", (REFIT, "direction-base1-k99", TAUGHT, 1)),
        ("trees-summaries-k99-s3", (REFIT, "trees-summaries-k99", TAUGHT, 3)),
        (
            "select-ft@0.0003-steps500-initialisation-k70-s2@selection",
            (SELECT, "select-ft@0.0003-steps500-initialisation-k70", TAUGHT, 2),
        ),
    ],
)
def test_an_answer_file_is_given_its_side_and_family(
    stem: str, expected: tuple[str, str, str, int]
) -> None:
    assert answer_detector(stem) == expected


@pytest.mark.parametrize(
    "stem",
    [
        # Scored on the validation months although fitted for selection.
        "select-ft@0.0003-steps500-initialisation-k70",
        # A pretraining read the selection months.
        "select-ft@0.0003-steps500-base1-k70-s1@selection",
    ],
)
def test_answers_that_would_leak_are_left_out(stem: str) -> None:
    assert answer_detector(stem) is None


def test_an_answer_no_family_claims_stops_the_run() -> None:
    with pytest.raises(ValueError, match="no family"):
        answer_detector("refit-something-new")


@pytest.mark.parametrize(
    ("candidate", "family"),
    [
        ("channel_local@reference=unit", DEPARTURES),
        ("pretext_error@error=largest,hidden=tail,reference=unit", OTHER),
        ("frozen_knn@distance=whitened,reference=unit", OTHER),
        ("frozen_knn@events=5,reference=unit", None),
        ("summaries_knn@events=5,head=trees", None),
    ],
)
def test_a_candidate_taught_a_few_events_is_left_out(candidate: str, family: str | None) -> None:
    assert candidate_family(candidate) == family


def test_a_unit_broken_in_two_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        month_slices(np.array([0, 0, 1, 1, 0]))


def test_a_rotation_turns_each_month_and_keeps_its_windows() -> None:
    structure, _ = month_world()
    slices = month_slices(structure.units)
    order = rotation_order(slices, np.random.default_rng(1))
    assert sorted(order.tolist()) == list(range(800))
    for start, length in slices:
        part = order[start : start + length]
        assert set(part.tolist()) == set(range(start, start + length))
        assert part[0] != start


def test_a_month_of_one_window_cannot_turn() -> None:
    order = rotation_order([(0, 1), (1, 3)], np.random.default_rng(1))
    assert order[0] == 0


@pytest.mark.parametrize("share", [0.0, 0.01, 0.1])
def test_the_fast_found_mask_agrees_with_the_tested_sweep(share: float) -> None:
    draws = np.random.default_rng(2)
    structure, ids = month_world()
    # A coarse grid leaves windows tied, across events and normal windows.
    scores = np.round(draws.standard_normal(800) * 2.0) / 2.0
    got = ids[found_mask(scores, structure.events, ids, share)].tolist()
    assert got == found_events(structure, scores, share).tolist()


def test_the_found_mask_counts_what_the_harness_counts() -> None:
    draws = np.random.default_rng(3)
    structure, ids = month_world()
    scores = draws.standard_normal(800)
    ranking = AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(
                unit=UnitKey(f"u{unit}"),
                position=position,
                ends_at=float(position),
            ),
            target=float(event),
            predicted=float(score),
        )
        for position, (unit, event, score) in enumerate(
            zip(structure.units.tolist(), structure.events.tolist(), scores.tolist(), strict=True)
        )
    )
    found = found_mask(scores, structure.events, ids, 0.05)
    assert found.mean() == pytest.approx(ranking.event_recall_at_false_alarm_share(0.05))


def only_event(structure: Structure, event: int, value: float = 9.0) -> np.ndarray:
    scores = np.linspace(-1.0, 0.0, len(structure.events))
    scores[structure.events == event] = value
    return scores


def test_a_detector_counts_an_event_found_in_at_least_half_its_repeats() -> None:
    structure, ids = month_world()
    always = only_event(structure, 1)
    twice = Detector("a", OTHER, (always, only_event(structure, 2), always))
    found = detector_found(twice, structure.events, ids, 0.01)
    assert found.tolist() == [True] + [False] * 7


def test_a_detector_of_two_repeats_needs_one_of_them() -> None:
    structure, ids = month_world()
    two = Detector("a", OTHER, (only_event(structure, 1), only_event(structure, 2)))
    assert detector_found(two, structure.events, ids, 0.01).tolist()[:2] == [True, True]


def test_an_event_is_filed_under_the_first_family_that_finds_it() -> None:
    named = categories(
        np.array([True, False, False, False]),
        np.array([True, True, False, False]),
        np.array([True, True, True, False]),
    )
    assert named == [DEPARTURES, OTHER, TAUGHT, NOTHING]


def test_detectors_of_another_family_do_not_count() -> None:
    structure, ids = month_world()
    detectors = [
        Detector("d", DEPARTURES, (only_event(structure, 1),)),
        Detector("t", TAUGHT, (only_event(structure, 2),)),
    ]
    found = family_found(detectors, OTHER, structure.events, ids, 0.01)
    assert not found.any()


def test_rotations_shared_by_detectors_do_not_multiply_duplicates() -> None:
    """Twenty copies of one detector add nothing to the union's chance level."""
    structure, ids = month_world()
    scores = np.random.default_rng(4).standard_normal(800)
    missed = np.zeros(8, dtype=bool)
    other = np.zeros(8, dtype=bool)
    one = excess_over_rotations(
        [Detector("a", OTHER, (scores,))],
        structure,
        ids,
        missed,
        other,
        share=0.05,
        rotations=200,
        draws=np.random.default_rng(5),
    )
    many = excess_over_rotations(
        [Detector(f"a{n}", OTHER, (scores,)) for n in range(20)],
        structure,
        ids,
        missed,
        other,
        share=0.05,
        rotations=200,
        draws=np.random.default_rng(5),
    )
    assert many.expected == pytest.approx(one.expected)


def test_the_excess_counts_only_the_events_the_departures_miss() -> None:
    structure, ids = month_world()
    departures = np.array([True, True, False, False, False, False, False, False])
    other = np.array([True, False, True, True, False, False, False, False])
    read = excess_over_rotations(
        [Detector("a", OTHER, (only_event(structure, 3),))],
        structure,
        ids,
        departures,
        other,
        share=0.01,
        rotations=50,
        draws=np.random.default_rng(6),
    )
    assert (read.missed, read.found) == (6, 2)
    assert read.per_event.shape == (8,)


def test_rotated_targets_keep_every_events_size() -> None:
    structure, _ = month_world()
    turned = rotated_targets(structure, np.random.default_rng(7))
    assert np.bincount(turned.events).tolist() == np.bincount(structure.events).tolist()
    assert not np.array_equal(turned.events, structure.events)


def test_the_fusion_takes_the_largest_rank_over_the_detectors() -> None:
    first = Detector("a", OTHER, (np.array([1.0, 2.0, 3.0, 4.0]),))
    second = Detector("b", OTHER, (np.array([4.0, 3.0, 2.0, 1.0]),))
    assert fused([first, second]).tolist() == pytest.approx([1.0, 0.75, 0.75, 1.0])


def test_month_reference_centres_and_scales_each_month() -> None:
    scores = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    read = month_referenced(scores, np.array([0, 0, 0, 1, 1, 1]))
    assert read[1] == read[4] == 0.0
    assert read[0] == pytest.approx(read[3])


def test_a_month_without_spread_is_only_centred() -> None:
    read = month_referenced(np.array([5.0, 5.0, 5.0]), np.array([0, 0, 0]))
    assert read.tolist() == [0.0, 0.0, 0.0]


def excess(found: int, expected: float, upper: float) -> Excess:
    return Excess(missed=30, found=found, expected=expected, upper=upper, per_event=np.zeros(1))


@pytest.mark.parametrize(
    ("read", "gain", "lower", "expected"),
    [
        (excess(20, 5.0, 9.0), 0.05, 0.01, True),
        # Beyond chance, but fewer events than the least gain asks of one detector.
        (excess(12, 5.0, 9.0), 0.05, 0.01, False),
        # As many events, but within what the rotations reach.
        (excess(20, 5.0, 20.0), 0.05, 0.01, False),
        # Room in the count, but a fusion that does not beat the departures alone.
        (excess(20, 5.0, 9.0), 0.05, -0.01, False),
        (excess(20, 5.0, 9.0), 0.01, 0.01, False),
    ],
)
def test_arm_b_needs_the_count_the_rotations_and_the_fusion(
    read: Excess, gain: float, lower: float, expected: bool
) -> None:
    assert reopens_perturbation_pretext(read, gain=gain, lower=lower, event_count=61) is expected


def test_the_annotated_channel_is_read_against_its_own_normal_windows() -> None:
    events = np.zeros(1000, dtype=np.int64)
    events[10:12] = 1
    departures = np.random.default_rng(8).uniform(0.0, 1.0, size=(1000, 3))
    departures[:, 2] *= 100.0
    departures[10, 0] = 5.0
    assert resolution_found(departures, events, 1, [0], 0.01)
    # On the other channel the same value is unremarkable.
    departures[10, 2] = 5.0
    departures[11, 2] = 0.0
    assert not resolution_found(departures, events, 1, [2], 0.01)


def test_a_constant_channel_finds_nothing() -> None:
    events = np.zeros(500, dtype=np.int64)
    events[3] = 1
    departures = np.zeros((500, 1))
    assert not resolution_found(departures, events, 1, [0], 0.01)


def test_the_alarms_are_read_per_channel_from_the_normal_windows() -> None:
    events = np.zeros(1000, dtype=np.int64)
    events[0] = 1
    departures = np.linspace(0.0, 1.0, 1000)[:, None] * np.array([[1.0, 10.0]])
    departures[0] = 1000.0
    alarms = channel_alarms(departures, events, [0, 1], 0.01)
    assert alarms[1] == pytest.approx(10.0 * alarms[0])
    assert alarms[0] < 1.0


def test_the_signal_is_the_largest_ratio_to_the_channels_alarm() -> None:
    values = np.array([[1.0, 4.0], [3.0, 2.0]])
    assert event_signal(values, np.array([1.0, 2.0])) == pytest.approx(3.0)


def test_a_channel_with_no_alarm_gives_no_signal() -> None:
    assert event_signal(np.array([[5.0]]), np.array([0.0])) == 0.0
    assert event_signal(np.empty((0, 1)), np.array([1.0])) == 0.0


def rotation_orders(length: int, offsets: range) -> list[np.ndarray]:
    return [(np.arange(length) + offset) % length for offset in offsets]


def test_a_strong_event_is_beyond_chance_however_many_windows_it_spans() -> None:
    """A long event passes the alarm under nearly every rotation; its size still tells."""
    draws = np.random.default_rng(9)
    events = np.zeros(2000, dtype=np.int64)
    events[100:400] = 1
    departures = draws.uniform(0.0, 1.0, size=(2000, 1))
    departures[250, 0] = 50.0
    alarms = channel_alarms(departures, events, [0], 0.01)
    orders = rotation_orders(2000, range(500, 1500))
    assert chance_of_signal(departures, events, 1, [0], orders, alarms) == 0.0


def test_a_long_event_of_noise_is_not_beyond_chance() -> None:
    draws = np.random.default_rng(10)
    events = np.zeros(2000, dtype=np.int64)
    events[100:400] = 1
    departures = draws.uniform(0.0, 1.0, size=(2000, 1))
    alarms = channel_alarms(departures, events, [0], 0.01)
    orders = rotation_orders(2000, range(1, 1000))
    assert chance_of_signal(departures, events, 1, [0], orders, alarms) > 0.05


def test_an_event_with_no_signal_is_never_beyond_chance() -> None:
    events = np.zeros(10, dtype=np.int64)
    events[0] = 1
    departures = np.zeros((10, 1))
    assert chance_of_signal(departures, events, 1, [0], [np.arange(10)], np.array([0.0])) == 1.0


def test_the_sources_are_read_from_their_file(tmp_path: Path) -> None:
    path = tmp_path / "sources.toml"
    path.write_text(
        'answers = ["a", "b"]\n'
        'truncated_from = "t"\ndepartures_from = "d"\nraw = "r"\n'
        '[[exports]]\npath = "e.csv"\ncampaigns = ["c1", "c2"]\n'
    )
    sources = Sources.read(path)
    assert sources.answers == (Path("a"), Path("b"))
    assert sources.exports == ((Path("e.csv"), ("c1", "c2")),)
    assert (sources.truncated_from, sources.departures_from, sources.raw) == (
        Path("t"),
        Path("d"),
        Path("r"),
    )


def row(channels: str, gained: str = "1") -> dict[str, str]:
    return {"gained": gained, "channels": channels}


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([row("14"), row("14"), row("21"), row("29")], True),
        # Four events on one set of channels are one piece of evidence.
        ([row("14"), row("14"), row("14"), row("14")], False),
        ([row("14"), row("21"), row("29")], False),
        ([row("14"), row("21"), row("29"), row("30", gained="0")], False),
    ],
)
def test_finer_cells_need_enough_events_on_enough_channels(
    rows: list[dict[str, str]], expected: bool
) -> None:
    assert queues_finer_cells(rows) is expected


def test_a_correlation_needs_values_that_vary() -> None:
    assert np.isnan(correlation(np.ones(20), np.arange(20.0)))
    assert correlation(np.arange(20.0), np.arange(20.0)) == pytest.approx(1.0)


def test_a_family_is_counted_under_each_rotation() -> None:
    structure, ids = month_world()
    detectors = [Detector("d", DEPARTURES, (only_event(structure, 1),))]
    identity = np.arange(800)
    counts = rotated_counts(detectors, DEPARTURES, structure, ids, [identity, identity])
    assert counts.tolist() == [1, 1]


def test_the_fusion_is_taken_once_per_repeat_of_the_varied_member() -> None:
    fixed = Detector("a", OTHER, (np.array([1.0, 2.0, 3.0]),))
    varied = Detector("b", OTHER, (np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])))
    first, second = fused_over_repeats([fixed], varied)
    assert first.tolist() == pytest.approx([1.0, 2.0 / 3.0, 1.0])
    assert second.tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 1.0])
