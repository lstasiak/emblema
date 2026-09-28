from statistics import fmean, stdev

import pytest

from emblema.evaluation.adapters.synthetic.known_ranking import KnownRanking


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
