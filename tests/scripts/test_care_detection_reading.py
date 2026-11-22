"""The ranking of the benchmark's excerpts: the event beside its year, the area and its chance."""

import numpy as np
import pytest

from scripts.care_detection_reading import (
    LEAST_SCORED,
    SPAN,
    Layout,
    Ranking,
    confirmed,
    largest_in_blocks,
    ranks_within,
    running_medians,
    shift_of,
    unusualness,
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


def test_equal_values_share_the_mean_of_their_ranks() -> None:
    assert ranks_within(np.array([[3.0, 1.0, 3.0, 2.0]])).tolist() == [[3.5, 1.0, 3.5, 2.0]]


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


def test_the_best_of_a_family_is_held_to_the_best_of_its_shuffles() -> None:
    draws = np.random.default_rng(8)
    labels = (np.arange(40) % 2 == 0).tolist()
    one = ranking(draws.standard_normal((1, 40)).tolist(), labels, ["A"] * 40)
    many = ranking(
        np.vstack([one.statistics, draws.standard_normal((30, 40)) * 1e-3]).tolist(),
        labels,
        ["A"] * 40,
    )
    best = float(many.areas().max())

    # The same best area is likelier by chance among thirty-one detectors than for one alone.
    alone = float((one.shuffled_areas(3000, seed=1)[:, 0] >= best).mean())
    assert many.best_beyond_chance(shuffles=3000)[1] > alone


def test_the_smallest_p_value_is_held_to_the_strictest_level() -> None:
    assert confirmed([0.012, 0.02]) == [True, True]
    assert confirmed([0.02, 0.02]) == [False, False]
    assert confirmed([0.03, 0.012]) == [False, True]


def test_the_layout_ranks_no_excerpt_lying_inside_a_fault() -> None:
    events = [
        Event("B", 27, 7, True, 0, 1, "", ""),
        Event("B", 87, 7, False, 0, 1, "", ""),
        Event("B", 34, 7, False, 0, 1, "", ""),
    ]

    layout = Layout.of(events)

    assert layout.anomaly.tolist() == [True, False]


def test_no_shift_is_an_area_of_a_half() -> None:
    assert shift_of(0.5) == 0.0
    assert shift_of(0.76) == pytest.approx(1.0, abs=0.01)
