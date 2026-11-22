"""The ranking of the benchmark's excerpts: the event beside its year, the area and its chance."""

from pathlib import Path

import numpy as np
import pytest

from scripts.care_detection_reading import (
    LEAST_SCORED,
    SPAN,
    Family,
    Layout,
    Ranking,
    Scored,
    area_rows,
    care_rows,
    excerpt_rows,
    flat,
    largest_in_blocks,
    missing_files,
    passes,
    power_rows,
    running_medians,
    shift_of,
    shuffled_labels,
    size_rows,
    unusualness,
    verdict_rows,
)
from scripts.care_excerpts import Event

YEAR = 2000
EVENT = (YEAR + 24, YEAR + 24 + 119)


def excerpt(seed: int, hours: int = YEAR + 200) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(hours)


def ranking(statistics: list[list[float]], anomaly: list[bool], farms: list[str]) -> Ranking:
    return Ranking(
        statistics=np.asarray(statistics, dtype=np.float64),
        anomaly=np.asarray(anomaly, dtype=np.bool_),
        farms=farms,
    )


def test_a_running_median_needs_its_whole_span() -> None:
    medians = running_medians(np.arange(20.0))

    assert np.isnan(medians[: SPAN - 1]).all()
    assert medians[SPAN - 1 :].tolist() == [5.5 + hour for hour in range(20 - SPAN + 1)]


def test_one_hours_spike_does_not_move_a_spans_median() -> None:
    scores = np.zeros(40)
    scores[20] = 1e6

    assert np.nanmax(running_medians(scores)) == 0.0


def test_a_span_with_too_few_scored_hours_has_no_median() -> None:
    scores = np.ones(SPAN)
    scores[: SPAN - LEAST_SCORED] = np.nan
    short = scores.copy()
    short[-1] = np.nan

    assert running_medians(scores)[-1] == 1.0
    assert np.isnan(running_medians(short)[-1])


def test_a_block_reads_only_the_spans_lying_inside_it() -> None:
    medians = running_medians(np.arange(30.0))

    largest = largest_in_blocks(medians, 15)

    # The first block's last span ends at hour 14, its median 8.5.
    assert largest[0] == 8.5
    assert len(largest) == 30 - 15 + 1


def test_blocks_with_no_scored_span_are_left_out() -> None:
    scores = np.ones(60)
    scores[:30] = np.nan

    assert len(largest_in_blocks(running_medians(scores), 15)) < 60 - 15 + 1


def test_an_event_that_stands_above_its_year_is_wholly_unusual() -> None:
    scores = excerpt(1)
    scores[EVENT[0] : EVENT[0] + 24] += 10.0

    assert unusualness(scores, fitted=YEAR, first=EVENT[0], last=EVENT[1]) == 1.0


def test_an_event_like_its_year_is_as_unusual_as_chance_whatever_its_length() -> None:
    for days in (2, 14, 60):
        last = YEAR + 24 * days - 1
        drawn = [
            unusualness(excerpt(seed, last + 1), fitted=YEAR, first=YEAR, last=last)
            for seed in range(60)
        ]

        assert np.mean([value for value in drawn if value is not None]) == pytest.approx(
            0.5, abs=0.12
        )


def test_nothing_outside_the_event_and_the_year_is_read() -> None:
    scores = excerpt(2)
    moved = scores.copy()
    moved[YEAR : EVENT[0]] += 50.0
    moved[EVENT[1] + 1 :] += 50.0

    assert unusualness(moved, fitted=YEAR, first=EVENT[0], last=EVENT[1]) == unusualness(
        scores, fitted=YEAR, first=EVENT[0], last=EVENT[1]
    )


def test_a_louder_year_makes_the_same_event_less_unusual() -> None:
    scores = excerpt(3)
    scores[EVENT[0] : EVENT[1] + 1] += 0.5
    louder = scores.copy()
    louder[:YEAR] += 0.5

    beside_louder = unusualness(louder, fitted=YEAR, first=EVENT[0], last=EVENT[1])
    beside_its_own = unusualness(scores, fitted=YEAR, first=EVENT[0], last=EVENT[1])

    assert beside_louder is not None
    assert beside_its_own is not None
    assert beside_louder < beside_its_own


def test_an_event_with_no_scored_span_has_no_statistic() -> None:
    scores = excerpt(4)
    scores[EVENT[0] : EVENT[1] + 1] = np.nan

    assert unusualness(scores, fitted=YEAR, first=EVENT[0], last=EVENT[1]) is None


def test_an_event_inside_the_fitted_year_is_refused() -> None:
    with pytest.raises(ValueError, match="do not follow"):
        unusualness(excerpt(5), fitted=YEAR, first=YEAR - 1, last=YEAR + 50)


def test_the_area_counts_pairs_inside_a_farm_only() -> None:
    # Every fault of the second farm lies below every quiet excerpt of the first, and above its
    # own farm's.
    ranked = ranking(
        [[10.0, 11.0, 12.0, 13.0, 1.0, 0.0, 2.0, 3.0]],
        [False, False, True, True, False, False, True, True],
        ["A"] * 4 + ["B"] * 4,
    )

    assert ranked.areas().tolist() == [1.0]


def test_the_area_is_a_half_for_ties_and_nought_for_a_ranking_upside_down() -> None:
    labels, farms = [False, False, True, True], ["A"] * 4

    assert ranking([[1.0, 1.0, 1.0, 1.0]], labels, farms).areas().tolist() == [0.5]
    assert ranking([[3.0, 4.0, 1.0, 2.0]], labels, farms).areas().tolist() == [0.0]
    # One fault ties with one quiet excerpt and stands above the other.
    assert ranking([[1.0, 2.0, 2.0, 3.0]], labels, farms).areas().tolist() == [0.875]


def test_excerpts_of_one_label_have_no_area() -> None:
    with pytest.raises(ValueError, match="no farm holds both"):
        ranking([[1.0, 2.0]], [True, True], ["A", "A"]).areas()


def test_an_event_as_flat_as_its_year_is_not_unusual() -> None:
    # Every block of the year reaches the event's largest median, which equals theirs.
    assert unusualness(np.zeros(YEAR + 200), fitted=YEAR, first=EVENT[0], last=EVENT[1]) == 0.0


def test_shuffled_labels_rank_at_chance_and_replay_from_their_seed() -> None:
    draws = np.random.default_rng(6)
    ranked = ranking(
        draws.standard_normal((3, 40)).tolist(),
        (draws.random(40) < 0.5).tolist(),
        ["A"] * 15 + ["B"] * 25,
    )

    shuffled = ranked.shuffled_areas(4000, seed=2)

    assert shuffled.shape == (4000, 3)
    assert shuffled.mean(axis=0).tolist() == pytest.approx([0.5] * 3, abs=0.01)
    assert np.array_equal(shuffled, ranked.shuffled_areas(4000, seed=2))


def test_a_shuffle_keeps_each_farms_count_of_faults() -> None:
    # One fault in a farm of three: a shuffle's area there is a third's multiple, never between.
    ranked = ranking([[1.0, 2.0, 3.0]], [False, False, True], ["A"] * 3)

    assert set(np.round(ranked.shuffled_areas(200, seed=1)[:, 0], 6)) == {0.0, 0.5, 1.0}


def layout() -> Layout:
    return Layout(
        farms=("A",) * 20 + ("B",) * 30,
        turbines=tuple(range(50)),
        anomaly=np.arange(50) % 2 == 0,
    )


def test_a_family_with_a_planted_fault_passes() -> None:
    planted = layout().drawn(
        np.random.default_rng(7),
        detectors=4,
        shifts=np.full(4, shift_of(0.9)),
        shared=0.5,
        turbine=0.0,
    )

    assert planted.best_beyond_chance(shuffles=2000)[1] < 0.01


def test_a_family_with_no_fault_passes_as_often_as_its_level() -> None:
    draws = np.random.default_rng(9)
    passed = [
        layout()
        .drawn(draws, detectors=4, shifts=np.zeros(4), shared=0.5, turbine=0.0)
        .best_beyond_chance(shuffles=499, seed=trial)[1]
        <= 0.05
        for trial in range(400)
    ]

    assert 0.02 < np.mean(passed) < 0.09


def test_the_best_of_a_family_is_held_to_the_best_of_each_shuffle() -> None:
    draws = np.random.default_rng(8)
    ranked = ranking(
        draws.standard_normal((6, 40)).tolist(), (np.arange(40) % 2 == 0).tolist(), ["A"] * 40
    )

    best, chance = ranked.best_beyond_chance(shuffles=500, seed=3)

    reached = (ranked.shuffled_areas(500, seed=3).max(axis=1) >= best).sum()
    assert best == ranked.areas().max()
    assert chance == (1 + reached) / 501
    # A single detector's shuffles reach the family's best less often than the family's do.
    alone = ranking(ranked.statistics[:1].tolist(), ranked.anomaly.tolist(), ["A"] * 40)
    assert (alone.shuffled_areas(500, seed=3)[:, 0] >= best).sum() < reached


def test_the_smaller_chance_is_held_to_the_level_over_both_verdicts() -> None:
    assert passes({"channel_local": 0.012, "normal_behaviour": 0.02}) == {
        "channel_local": True,
        "normal_behaviour": True,
    }
    assert passes({"channel_local": 0.02, "normal_behaviour": 0.02}) == {
        "channel_local": False,
        "normal_behaviour": False,
    }
    assert passes({"channel_local": 0.03, "normal_behaviour": 0.012}) == {
        "channel_local": False,
        "normal_behaviour": True,
    }


def test_a_verdict_read_alone_keeps_the_level_both_would_give_it() -> None:
    assert passes({"channel_local": 0.02}) == {"channel_local": False}
    assert passes({"channel_local": 0.012}) == {"channel_local": True}


def test_the_power_table_holds_what_a_family_must_reach_and_how_often_it_does() -> None:
    rows = power_rows(layout(), draws=200, seed=1, areas=(0.6, 0.9))

    assert len(rows) == 2 * (1 + 4 * 2)
    assert all(len(row) == 5 + 2 for row in rows)
    bars = {(row[0], row[1], row[2]): float(row[3]) for row in rows}
    assert 0.5 < bars["1", "0.5", "2.50 %"] < bars["1", "0.5", "1.25 %"] < 1.0
    assert bars["12", "0.5", "1.25 %"] > bars["1", "0.5", "1.25 %"]
    every = next(row for row in rows if row[:3] == ["12", "0.5", "2.50 %"] and row[4] == "every")
    assert float(every[5]) < float(every[6])


def test_the_size_table_reads_both_levels_at_each_share_of_a_turbines_variance() -> None:
    rows = size_rows(layout(), draws=5, seed=1)

    assert [row[0] for row in rows] == ["0.0", "0.2", "0.4"]
    assert all(len(row) == 3 and row[1].endswith(" %") for row in rows)


def test_the_layout_ranks_no_excerpt_lying_inside_a_fault() -> None:
    events = [
        Event("B", 27, 7, True, 0, 1),
        Event("B", 87, 7, False, 0, 1),
        Event("B", 34, 7, False, 0, 1),
    ]

    layout = Layout.of(events)

    assert layout.anomaly.tolist() == [True, False]


def test_no_shift_is_an_area_of_a_half() -> None:
    assert shift_of(0.5) == 0.0
    assert shift_of(0.76) == pytest.approx(1.0, abs=0.01)


# --- the reading of score files ---------------------------------------------------------------

FITTED, SCORED_HOURS = 1500, 160
FIRST, LAST = FITTED + 60, FITTED + 149
DETECTORS = (
    "availability",
    "channel_local/max-window-6h",
    "channel_local/mean-window-6h",
    "normal_behaviour/max",
    "normal_behaviour/mean",
)


def files(
    seed: int, lift: float = 0.0, names: tuple[str, ...] = DETECTORS, last: int = LAST
) -> list[dict]:
    """An excerpt's two score files, its event's hours lifted above the year's."""
    draws = np.random.default_rng(seed)
    hours = FITTED + SCORED_HOURS
    scores = draws.standard_normal((len(names), hours))
    scores[:, FIRST : last + 1] += lift
    rows = np.arange(hours * 6)
    placed = {"fitted": np.asarray(FITTED), "first": np.asarray(FIRST), "last": np.asarray(last)}
    split = sum(not name.startswith("normal_behaviour") for name in names)
    return [
        {
            "arms": np.asarray(names[:split]),
            "scores": scores[:split].astype(np.float32),
            "row_hour": rows // 6,
            "row_fitted": rows < FITTED * 6,
            "row_normal": np.ones(len(rows), dtype=np.bool_),
            "row_id": rows,
            **placed,
        },
        {"arms": np.asarray(names[split:]), "scores": scores[split:].astype(np.float32), **placed},
    ]


def event(
    number: int, *, farm: str = "A", turbine: int = 1, anomaly: bool = True, last: int = LAST
) -> Event:
    return Event(farm, number, turbine, anomaly, FIRST * 6, last * 6 + 5)


def excerpts(lift: float = 3.0, count: int = 24, turbines: int = 6) -> list[Scored]:
    """Excerpts of two farms, every second with a fault, its event lifted by ``lift``."""
    return [
        Scored.joined(
            event(number, farm="AB"[number % 2], turbine=number % turbines, anomaly=number % 4 < 2),
            files(number, lift if number % 4 < 2 else 0.0),
        )
        for number in range(count)
    ]


def test_an_excerpts_files_are_joined_by_detector_and_its_event_placed_in_the_scored_rows() -> None:
    joined = Scored.joined(event(1), files(1))

    assert set(joined.series) == set(DETECTORS)
    assert (joined.fitted, joined.first, joined.last) == (FITTED, FIRST, LAST)
    assert len(joined.row_hour) == SCORED_HOURS * 6
    assert joined.rows == (60 * 6, 150 * 6 - 1)


def test_files_that_place_an_excerpts_hours_differently_are_refused() -> None:
    first, second = files(1)
    second["first"] = np.asarray(FIRST + 1)

    with pytest.raises(ValueError, match="place its hours differently"):
        Scored.joined(event(1), [first, second])


def test_an_alarm_is_set_on_the_fitted_year_alone() -> None:
    first, second = files(2, lift=50.0)
    joined = Scored.joined(event(2), [first, second])

    flagged = joined.flagged("normal_behaviour/max", 0.02)

    # Every hour of the event stands above the year's 98th percentile, and few others do.
    assert flagged[60 * 6 : 150 * 6].all()
    assert flagged[: 60 * 6].mean() < 0.15


def test_an_hour_that_is_not_scored_is_not_flagged() -> None:
    first, second = files(3, lift=50.0)
    first["scores"][1, FIRST : FIRST + 10] = np.nan

    flagged = Scored.joined(event(3), [first, second]).flagged(DETECTORS[1], 0.02)

    assert not flagged[60 * 6 : 70 * 6].any()
    assert flagged[70 * 6 : 150 * 6].all()


def test_a_detector_that_scored_no_hour_of_the_year_flags_nothing() -> None:
    first, second = files(3, lift=50.0)
    first["scores"][1, :FITTED] = np.nan

    joined = Scored.joined(event(3), [first, second])

    assert not joined.flagged(DETECTORS[1], 0.02).any()
    assert joined.statistic(DETECTORS[1]) == 0.0


def test_the_events_length_stands_in_for_a_detector_that_reads_nothing_else() -> None:
    assert Scored.joined(event(1), files(1)).statistic("event_length") == LAST - FIRST + 1


def test_a_trials_labels_are_shuffled_inside_each_farm_and_replay() -> None:
    read = excerpts()

    shuffled = shuffled_labels(read, seed=4)

    for seed in range(8):
        for farm in "AB":
            before = sum(e.event.anomaly for e in read if e.event.farm == farm)
            drawn = shuffled_labels(read, seed=seed)
            assert sum(e.event.anomaly for e in drawn if e.event.farm == farm) == before
    assert [e.event.anomaly for e in shuffled] != [e.event.anomaly for e in read]
    assert [e.event.anomaly for e in shuffled] == [
        e.event.anomaly for e in shuffled_labels(read, seed=4)
    ]


def test_an_event_a_detector_cannot_read_takes_the_lowest_statistic_and_stays_ranked() -> None:
    read = excerpts(count=8)
    unscored_first, unscored_second = files(99, lift=50.0)
    unscored_first["scores"][1, FIRST : LAST + 1] = np.nan
    read.append(Scored.joined(event(99, farm="A"), [unscored_first, unscored_second]))

    family = Family.of(DETECTORS[1:3], read)

    assert len(family.excerpts) == 9
    assert family.unread == 1
    assert family.ranking.statistics[:, -1].tolist() == [0.0, 1.0]
    assert not read[-1].read_by(DETECTORS[1])


def test_the_excerpt_lying_inside_a_fault_is_ranked_by_no_family() -> None:
    read = excerpts(count=8)
    read.append(Scored.joined(Event("B", 87, 7, False, FIRST * 6, LAST * 6), files(87)))

    assert len(Family.of(DETECTORS[1:3], read).excerpts) == 8
    assert len(Family.of(DETECTORS[1:3], read, every=True).excerpts) == 9


def test_a_verdicts_best_is_read_again_with_the_excerpt_inside_a_fault_ranked() -> None:
    read = excerpts(lift=3.0, count=8)
    # A quiet excerpt that looks like a fault: ranked, it costs the best detector pairs.
    read.append(Scored.joined(Event("B", 87, 7, False, FIRST * 6, LAST * 6), files(87, lift=50.0)))

    row = next(row for row in verdict_rows(read, 500, seed=1) if row.family == "normal_behaviour")

    assert row.excerpts == 8
    assert row.best_area == 1.0
    assert row.with_every_excerpt < 1.0


def test_a_planted_fault_passes_its_familys_verdict_and_noise_does_not() -> None:
    planted = {row.family: row for row in verdict_rows(excerpts(lift=3.0), 2000, seed=1)}
    quiet = {row.family: row for row in verdict_rows(excerpts(lift=0.0), 2000, seed=1)}

    # The answer with no model is one detector and no family: its area stands in the other table.
    assert set(planted) == {"channel_local", "normal_behaviour"}
    assert planted["channel_local"].confirmed is True
    assert planted["normal_behaviour"].confirmed is True
    assert planted["channel_local"].best_area > 0.9
    assert planted["channel_local"].best in DETECTORS[1:3]
    assert quiet["channel_local"].confirmed is False


def test_every_detector_has_its_own_area_inside_an_interval_over_turbines_drawn_again() -> None:
    # A lift small enough that the faults stand above the quiet excerpts often, not always.
    rows = {row.detector: row for row in area_rows(excerpts(lift=0.3), 500, seed=1)}

    assert set(rows) == {*DETECTORS, "event_length"}
    for name in DETECTORS[1:]:
        assert 0.5 < rows[name].area < 1.0
        assert rows[name].low < rows[name].area < rows[name].high
        assert rows[name].high - rows[name].low > 0.05
        assert set(rows[name].by_farm) == {"A", "B"}


def test_the_events_length_alone_is_ranked_beside_the_detectors() -> None:
    read = [
        Scored.joined(
            event(number, anomaly=number % 2 == 0, turbine=number, last=LAST - 40 * (number % 2)),
            files(number, last=LAST - 40 * (number % 2)),
        )
        for number in range(8)
    ]

    rows = {row.detector: row for row in area_rows(read, 200, seed=1)}

    # Every fault's event is the longer one.
    assert rows["event_length"].area == 1.0


def test_an_event_beyond_every_block_of_its_year_is_counted() -> None:
    rows = {row.detector: row for row in area_rows(excerpts(lift=50.0, count=8), 200, seed=1)}

    assert rows["normal_behaviour/max"].beyond_the_year == 4
    assert rows["normal_behaviour/max"].unread == 0


def test_a_tables_row_is_written_with_its_farms_areas_as_columns() -> None:
    (row, *_) = area_rows(excerpts(lift=1.0, count=8), 200, seed=1)

    assert {"detector", "area", "own_chance", "area_A", "area_B"} <= set(flat(row))


def lifted(read: list[Scored], name: str) -> list[Scored]:
    """The excerpts with one detector's scores lifted through every fault's event."""
    out = []
    for excerpt in read:
        scores = excerpt.series[name].copy()
        if excerpt.event.anomaly:
            scores[FIRST : LAST + 1] += 3.0
        out.append(
            Scored.joined(
                excerpt.event,
                [
                    {
                        "arms": np.asarray(list(excerpt.series)),
                        "scores": np.stack(
                            [
                                scores if own == name else excerpt.series[own]
                                for own in excerpt.series
                            ]
                        ),
                        "row_hour": np.arange((FITTED + SCORED_HOURS) * 6) // 6,
                        "row_fitted": np.arange((FITTED + SCORED_HOURS) * 6) < FITTED * 6,
                        "row_normal": np.ones((FITTED + SCORED_HOURS) * 6, dtype=np.bool_),
                        "row_id": np.arange((FITTED + SCORED_HOURS) * 6),
                        "fitted": np.asarray(FITTED),
                        "first": np.asarray(FIRST),
                        "last": np.asarray(LAST),
                    }
                ],
            )
        )
    return out


def test_two_detectors_alike_differ_by_nothing_and_a_better_one_by_more_than_chance() -> None:
    # A turbine each: six turbines swap in 64 ways only, and no chance falls below one in 32.
    read = lifted(excerpts(lift=0.0, count=40, turbines=40), DETECTORS[3])

    alike = Family.of([DETECTORS[1], DETECTORS[1]], read).difference(swaps=500)
    better = Family.of([DETECTORS[3], DETECTORS[1]], read).difference(swaps=2000)
    worse = Family.of([DETECTORS[1], DETECTORS[3]], read).difference(swaps=2000)

    assert alike == (0.0, 1.0)
    assert better[0] > 0.3
    assert better[1] < 0.01
    # The chance is two-sided: the worse detector's deficit is as unlikely as the better's lead.
    assert worse == (-better[0], better[1])


def test_two_detectors_are_swapped_in_all_of_a_turbines_excerpts_at_once() -> None:
    # Two turbines swap in four ways, two of which give the difference read or its mirror, so
    # no difference, however large, is rarer than one in two.
    read = lifted(excerpts(lift=0.0, count=40, turbines=2), DETECTORS[3])

    difference, chance = Family.of([DETECTORS[3], DETECTORS[1]], read).difference(swaps=2000)

    assert difference > 0.3
    assert chance > 0.4


def test_the_benchmarks_score_is_read_at_every_alarm_for_every_detector() -> None:
    rows = care_rows(excerpts(lift=50.0, count=8))

    # Every detector but the answer with no model, which counts rows the score leaves out.
    assert len(rows) == (len(DETECTORS) - 1) * 5
    loud = next(
        row.reading
        for row in rows
        if row.detector == "normal_behaviour/max" and row.alarm_share == 0.005
    )
    assert (loud.found, loud.falsely_found) == (4, 0)
    # The event starts sixty hours into the scored window: placed at the window's start, a
    # third of its rows would be unflagged and the coverage about 0.9.
    assert loud.coverage > 0.97
    assert loud.score > 0.9
    assert {"detector", "alarm_share", "coverage", "score"} <= set(flat(rows[0]))


def test_the_benchmarks_score_leaves_out_the_rows_not_of_normal_operation() -> None:
    read = excerpts(lift=0.0, count=8)
    loud_first, loud_second = files(0, lift=50.0)
    loud_first["row_normal"][FIRST * 6 : (LAST + 1) * 6] = False
    read[2] = Scored.joined(read[2].event, [loud_first, loud_second])

    quiet = next(
        row.reading
        for row in care_rows(read)
        if row.detector == "normal_behaviour/max" and row.alarm_share == 0.005
    )

    # The quiet excerpt's flagged hours are all outside normal operation and cost it nothing.
    assert not read[2].event.anomaly
    assert quiet.accuracy > 0.98


def test_a_reading_names_the_score_files_it_lacks(tmp_path: Path) -> None:
    events = [event(1), event(2)]
    for name in ("departures-A-1.npz", "behaviour-A-1.npz", "departures-A-2.npz"):
        (tmp_path / name).touch()

    assert missing_files(events, tmp_path) == [tmp_path / "behaviour-A-2.npz"]


def test_the_excerpts_table_holds_every_detectors_statistic() -> None:
    (row, *_) = excerpt_rows(excerpts(count=4))

    assert {"farm", "event", "turbine", "anomaly", "ranked", "event_hours", *DETECTORS} <= set(row)
    assert row["event_hours"] == LAST - FIRST + 1
