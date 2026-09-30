from statistics import fmean, stdev

import pytest

from emblema.evaluation.adapters.synthetic.known_ranking import KnownRanking
from emblema.evaluation.domain.exceptions import InvalidKnownRankingError


def test_the_generator_states_the_areas_it_draws() -> None:
    paired = KnownRanking(control_auroc=0.80, candidate_auroc=0.85, units=4000, positives=560)

    drawn = paired.repeats(3, seed=1)

    assert 1.0 - drawn.error_control == pytest.approx(0.80, abs=0.02)
    assert 1.0 - drawn.error_candidate == pytest.approx(0.85, abs=0.02)


def test_without_a_spread_of_repeats_every_repeat_ranks_near_the_stated_area() -> None:
    known = KnownRanking(control_auroc=0.80, candidate_auroc=0.80, units=4000, positives=560)

    areas = [1.0 - shortfall for shortfall in known.repeats(20, seed=2).per_repeat_control()]

    assert stdev(areas) < 0.015


def test_a_spread_of_repeats_scatters_each_repeat_around_the_stated_area() -> None:
    known = KnownRanking(
        control_auroc=0.80,
        candidate_auroc=0.80,
        units=4000,
        positives=560,
        repeat_spread=0.05,
    )

    areas = [1.0 - shortfall for shortfall in known.repeats(20, seed=2).per_repeat_control()]

    assert 0.03 < stdev(areas) < 0.08
    assert fmean(areas) == pytest.approx(0.80, abs=0.03)


def test_a_spread_of_zero_draws_what_the_generator_drew_before_it_had_one() -> None:
    plain = KnownRanking(control_auroc=0.75, candidate_auroc=0.80)
    zero = KnownRanking(control_auroc=0.75, candidate_auroc=0.80, repeat_spread=0.0)

    assert plain.repeats(3, seed=7) == zero.repeats(3, seed=7)


def test_a_wide_spread_near_a_perfect_ranking_never_draws_an_area_beyond_it() -> None:
    known = KnownRanking(control_auroc=0.95, candidate_auroc=0.95, repeat_spread=0.1)

    areas = [1.0 - shortfall for shortfall in known.repeats(50, seed=3).per_repeat_control()]

    assert all(0.0 < area <= 1.0 for area in areas)


def test_the_stated_area_is_the_mean_over_many_repeats_even_under_a_wide_spread() -> None:
    known = KnownRanking(
        control_auroc=0.85, candidate_auroc=0.85, units=1000, positives=140, repeat_spread=0.1
    )

    areas = [1.0 - shortfall for shortfall in known.repeats(300, seed=5).per_repeat_control()]

    # Scattering the quantile without widening it first would average about 0.83 here.
    assert fmean(areas) == pytest.approx(0.85, abs=0.01)


def test_answers_cut_into_levels_take_only_that_many_values() -> None:
    known = KnownRanking(control_auroc=0.80, candidate_auroc=0.82, answer_levels=5)

    drawn = known.paired(seed=4)

    for ranking in (*drawn.control, *drawn.candidate):
        # The closing window of each group of ties marks one distinct answer.
        assert sum(ranking.closes_tie) == 5


def test_answers_cut_into_levels_reach_the_area_the_cut_leaves_and_not_the_stated_one() -> None:
    known = KnownRanking(
        control_auroc=0.80, candidate_auroc=0.80, units=4000, positives=560, answer_levels=4
    )

    areas = [1.0 - shortfall for shortfall in known.repeats(30, seed=6).per_repeat_control()]

    assert known.reachable_area(0.80) < 0.78
    assert fmean(areas) == pytest.approx(known.reachable_area(0.80), abs=0.005)


def test_the_true_reduction_under_levels_is_the_difference_of_the_areas_reachable() -> None:
    known = KnownRanking(control_auroc=0.80, candidate_auroc=0.82, answer_levels=4)

    assert known.true_reduction == known.reachable_area(0.82) - known.reachable_area(0.80)
    assert known.true_reduction != pytest.approx(0.02, abs=1e-3)
    assert (
        KnownRanking(control_auroc=0.80, candidate_auroc=0.80, answer_levels=4).true_reduction
        == 0.0
    )


def test_many_levels_come_close_to_answers_that_never_tie() -> None:
    known = KnownRanking(control_auroc=0.80, candidate_auroc=0.80, answer_levels=200)

    assert known.reachable_area(0.80) == pytest.approx(0.80, abs=0.002)


@pytest.mark.parametrize(
    ("levels", "spread"), [(1, 0.0), (-2, 0.0), (5, 0.01)], ids=["one", "negative", "with-spread"]
)
def test_levels_that_cannot_rank_or_cannot_stay_fixed_are_refused(
    levels: int, spread: float
) -> None:
    with pytest.raises(InvalidKnownRankingError):
        KnownRanking(
            control_auroc=0.80, candidate_auroc=0.80, answer_levels=levels, repeat_spread=spread
        )
