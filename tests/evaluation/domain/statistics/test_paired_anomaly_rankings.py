import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from tests.evaluation.support import prediction

# Four units, an event on the second and the fourth; the control finds the second event only
# after every normal window, the candidate finds both first.
WINDOWS = (("a", 0, 0.0), ("b", 0, 1.0), ("c", 0, 0.0), ("d", 0, 2.0))
CONTROL = (0.9, 0.1, 0.8, 0.7)
CANDIDATE = (0.1, 0.9, 0.2, 0.8)


def ranking(scores: tuple[float, ...]) -> AnomalyRanking:
    return AnomalyRanking.of(
        prediction(unit, position, event, score)
        for (unit, position, event), score in zip(WINDOWS, scores, strict=True)
    )


PAIRED = PairedAnomalyRankings(control=(ranking(CONTROL),), candidate=(ranking(CANDIDATE),))


def test_the_error_of_a_side_is_its_shortfall_from_a_perfect_area() -> None:
    # The control: d admitted at precision 1/2 (two normal windows above it ... no: d third
    # with a and c above it), b last at 2/4; the candidate finds both first.
    assert PAIRED.error_candidate == pytest.approx(0.0)
    assert PAIRED.error_control == pytest.approx(1.0 - (1 / 3 + 2 / 4) / 2)
    assert PAIRED.reduction == pytest.approx(PAIRED.error_control)
    assert PAIRED.relative_reduction == pytest.approx(1.0)


def test_repeats_pool_by_the_mean_of_their_areas() -> None:
    paired = PairedAnomalyRankings(
        control=(ranking(CONTROL), ranking(CANDIDATE)), candidate=(ranking(CANDIDATE),)
    )

    assert paired.error_control == pytest.approx(PAIRED.error_control / 2)
    assert paired.per_repeat_control() == pytest.approx((PAIRED.error_control, 0.0))
    assert paired.per_repeat_candidate() == (0.0,)


def test_the_strata_are_the_units_holding_an_event_and_the_rest() -> None:
    assert PAIRED.strata == ((1, 3), (0, 2))


def test_a_resample_counts_a_unit_as_often_as_it_is_picked() -> None:
    # Every event unit picked and both normal units twice: the control's event b is admitted
    # after four normal windows' weight.
    reduction = PAIRED.reduction_over([1, 3, 0, 0, 2, 2])

    control = 1.0 - (1 / 5 + 2 / 6) / 2
    assert reduction == pytest.approx(control - 0.0)


def test_the_bootstrap_draws_each_stratum_on_its_own() -> None:
    difference = PairedUnitBootstrap(resamples=50, seed=1).compare(PAIRED)

    assert difference.reduction == pytest.approx(PAIRED.reduction)
    assert 0.0 < difference.interval.low <= difference.interval.high <= 1.0


def test_rankings_over_other_units_or_events_do_not_pair() -> None:
    other_units = AnomalyRanking.of([prediction("a", 0, 0.0, 0.1), prediction("e", 0, 1.0, 0.9)])
    other_events = AnomalyRanking.of(
        prediction(unit, position, 3.0 if event else 0.0, score)
        for (unit, position, event), score in zip(WINDOWS, CONTROL, strict=True)
    )

    with pytest.raises(InvalidPairedAnomalyRankingsError, match="units"):
        PairedAnomalyRankings(control=(ranking(CONTROL),), candidate=(other_units,))
    with pytest.raises(InvalidPairedAnomalyRankingsError, match="events"):
        PairedAnomalyRankings(control=(ranking(CONTROL),), candidate=(other_events,))
    with pytest.raises(InvalidPairedAnomalyRankingsError, match="repeat on each side"):
        PairedAnomalyRankings(control=(), candidate=(ranking(CONTROL),))
