import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_event_recalls import PairedEventRecalls
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from tests.evaluation.support import prediction

# Four units, an event on the second and the fourth; the control ranks a normal window first
# and the event b last, the candidate ranks both events first.
WINDOWS = (("a", 0, 0.0), ("b", 0, 1.0), ("c", 0, 0.0), ("d", 0, 2.0))
CONTROL = (0.9, 0.1, 0.8, 0.95)
CANDIDATE = (0.1, 0.9, 0.2, 0.8)


def ranking(scores: tuple[float, ...]) -> AnomalyRanking:
    return AnomalyRanking.of(
        prediction(unit, position, event, score)
        for (unit, position, event), score in zip(WINDOWS, scores, strict=True)
    )


RANKINGS = PairedAnomalyRankings(control=(ranking(CONTROL),), candidate=(ranking(CANDIDATE),))


def test_the_error_of_a_side_is_the_share_of_events_it_misses_at_the_alarm_rate() -> None:
    # With no normal window flagged the control finds d alone, the candidate both events.
    paired = PairedEventRecalls(rankings=RANKINGS, share=0.0)

    assert (paired.error_control, paired.error_candidate) == (0.5, 0.0)
    assert paired.reduction == 0.5
    assert paired.relative_reduction == 1.0
    assert paired.per_repeat_control() == (0.5,)
    assert paired.per_repeat_candidate() == (0.0,)
    # Flagging every normal window finds every event on either side.
    assert PairedEventRecalls(rankings=RANKINGS, share=1.0).reduction == 0.0
    assert PairedEventRecalls(rankings=RANKINGS, share=1.0).relative_reduction is None


def test_the_draws_and_strata_are_the_areas() -> None:
    paired = PairedEventRecalls(rankings=RANKINGS, share=0.0)

    assert (paired.units, paired.strata) == (RANKINGS.units, RANKINGS.strata)
    # Without b the control misses nothing either; with b twice it misses two events of three.
    assert paired.reduction_over([3, 0, 2]) == 0.0
    assert paired.reductions_over([[1, 1, 3, 0], [1, 3, 0, 2]]) == pytest.approx([2 / 3, 0.5])


def test_repeats_pool_by_the_mean_of_their_shares() -> None:
    paired = PairedEventRecalls(
        rankings=PairedAnomalyRankings(
            control=(ranking(CONTROL), ranking(CANDIDATE)), candidate=(ranking(CANDIDATE),)
        ),
        share=0.0,
    )

    assert paired.error_control == 0.25


def test_the_bootstrap_reads_the_events_found_as_it_reads_an_area() -> None:
    difference = PairedUnitBootstrap(resamples=200, seed=1).compare(
        PairedEventRecalls(rankings=RANKINGS, share=0.0)
    )

    assert difference.reduction == 0.5
    assert 0.0 <= difference.interval.low <= difference.interval.high <= 1.0


@pytest.mark.parametrize("share", [-0.01, 1.5, float("nan")])
def test_a_share_outside_zero_to_one_is_refused(share: float) -> None:
    with pytest.raises(InvalidPairedAnomalyRankingsError, match=r"\[0, 1\]"):
        PairedEventRecalls(rankings=RANKINGS, share=share)
