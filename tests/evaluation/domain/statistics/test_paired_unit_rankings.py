import pytest

from emblema.evaluation.adapters.synthetic.known_ranking import KnownRanking
from emblema.evaluation.domain.exceptions import InvalidPairedUnitRankingsError
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from tests.evaluation.support import prediction


def ranked(*rows: tuple[str, float, float]) -> WindowRanking:
    return WindowRanking.of(
        prediction(unit, 0, target, predicted) for unit, target, predicted in rows
    )


CONTROL = ranked(("a", 0.0, 0.2), ("b", 1.0, 0.1), ("c", 0.0, 0.3), ("d", 1.0, 0.9))
CANDIDATE = ranked(("a", 0.0, 0.2), ("b", 1.0, 0.5), ("c", 0.0, 0.3), ("d", 1.0, 0.9))


def test_the_reduction_is_the_area_the_candidate_gains() -> None:
    paired = PairedUnitRankings(control=(CONTROL,), candidate=(CANDIDATE,))

    assert paired.error_control == pytest.approx(0.5)
    assert paired.error_candidate == 0.0
    assert paired.reduction == pytest.approx(0.5)
    assert paired.relative_reduction == pytest.approx(1.0)


def test_repeats_are_pooled_by_the_mean_of_their_areas() -> None:
    paired = PairedUnitRankings(control=(CONTROL, CANDIDATE), candidate=(CANDIDATE,))

    assert paired.error_control == pytest.approx(0.25)
    assert paired.per_repeat_control() == pytest.approx((0.5, 0.0))


def test_the_units_are_drawn_in_two_strata_by_whether_they_hold_a_positive() -> None:
    paired = PairedUnitRankings(control=(CONTROL,), candidate=(CANDIDATE,))

    assert paired.strata == ((1, 3), (0, 2))


def test_a_draw_that_counts_a_unit_twice_weighs_it_twice() -> None:
    paired = PairedUnitRankings(control=(CONTROL,), candidate=(CANDIDATE,))

    # Unit b drawn twice and d not at all: the control ranks both copies of b under both
    # negatives, the candidate over them.
    assert paired.reduction_over([0, 1, 1, 2]) == pytest.approx(1.0)


def test_rankings_of_other_units_do_not_pair() -> None:
    other = ranked(("a", 0.0, 0.2), ("b", 1.0, 0.1), ("e", 0.0, 0.3), ("d", 1.0, 0.9))

    with pytest.raises(InvalidPairedUnitRankingsError, match="units"):
        PairedUnitRankings(control=(CONTROL,), candidate=(other,))


def test_rankings_of_other_outcomes_do_not_pair() -> None:
    other = ranked(("a", 1.0, 0.2), ("b", 1.0, 0.1), ("c", 0.0, 0.3), ("d", 0.0, 0.9))

    with pytest.raises(InvalidPairedUnitRankingsError, match="outcomes"):
        PairedUnitRankings(control=(CONTROL,), candidate=(other,))


def test_a_side_without_a_repeat_is_refused() -> None:
    with pytest.raises(InvalidPairedUnitRankingsError, match="repeat on each side"):
        PairedUnitRankings(control=(), candidate=(CANDIDATE,))


def test_a_control_that_ranks_perfectly_has_no_shortfall_to_share() -> None:
    paired = PairedUnitRankings(control=(CANDIDATE,), candidate=(CANDIDATE,))

    assert paired.reduction == 0.0
    assert paired.relative_reduction is None


def test_the_generator_states_the_areas_it_draws() -> None:
    paired = KnownRanking(control_auroc=0.80, candidate_auroc=0.85, units=4000, positives=560)

    drawn = paired.repeats(3, seed=1)

    assert 1.0 - drawn.error_control == pytest.approx(0.80, abs=0.02)
    assert 1.0 - drawn.error_candidate == pytest.approx(0.85, abs=0.02)


def test_every_resample_keeps_the_count_of_each_stratum() -> None:
    paired = KnownRanking(control_auroc=0.80, candidate_auroc=0.85).paired(seed=3)
    positives, negatives = paired.strata

    bootstrap = PairedUnitBootstrap(resamples=200, seed=1)

    # A resample that dropped every positive would raise; two hundred of them do not.
    assert len(bootstrap.reductions(paired)) == 200
    assert (len(positives), len(negatives)) == (56, 344)


def test_a_true_gain_in_area_is_found_with_its_whole_interval_above_zero() -> None:
    known = KnownRanking(control_auroc=0.75, candidate_auroc=0.85, units=1600, positives=224)

    compared = PairedUnitBootstrap(resamples=500, seed=1).compare(known.repeats(3, seed=5))

    assert compared.reduction == pytest.approx(known.true_reduction, abs=0.03)
    assert compared.interval.above_zero
    assert compared.confirms(0.05, ThresholdKind.ABSOLUTE)


def test_no_true_gain_in_area_leaves_zero_inside_the_interval() -> None:
    known = KnownRanking(control_auroc=0.80, candidate_auroc=0.80, units=1600, positives=224)

    compared = PairedUnitBootstrap(resamples=500, seed=1).compare(known.repeats(3, seed=5))

    assert abs(compared.reduction) < 0.02
    assert not compared.distinguishable
