import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedAnomalyRankingsError
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.statistics.anomaly_interaction import AnomalyInteraction
from emblema.evaluation.domain.statistics.paired_anomaly_rankings import PairedAnomalyRankings
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from tests.evaluation.support import prediction

# Four units, an event on the second and the fourth.
WINDOWS = (("a", 0, 0.0), ("b", 0, 1.0), ("c", 0, 0.0), ("d", 0, 2.0))
LAST = (0.9, 0.1, 0.8, 0.2)
MIDDLE = (0.9, 0.5, 0.1, 0.6)
FIRST = (0.1, 0.9, 0.2, 0.8)


def ranking(scores: tuple[float, ...]) -> AnomalyRanking:
    return AnomalyRanking.of(
        prediction(unit, position, event, score)
        for (unit, position, event), score in zip(WINDOWS, scores, strict=True)
    )


def paired(control: tuple[float, ...], candidate: tuple[float, ...]) -> PairedAnomalyRankings:
    return PairedAnomalyRankings(control=(ranking(control),), candidate=(ranking(candidate),))


def test_the_interaction_is_the_taught_pairs_reduction_less_the_untaught_pairs() -> None:
    # Untaught, both sides find the events last; taught, the control finds them in the middle
    # and the candidate first: the candidate gains more from the teaching than the control.
    untaught = paired(LAST, LAST)
    taught = paired(MIDDLE, FIRST)
    interaction = AnomalyInteraction(taught=taught, untaught=untaught)

    assert untaught.reduction == pytest.approx(0.0)
    assert interaction.reduction == pytest.approx(taught.reduction)
    assert interaction.relative_reduction is None
    assert interaction.strata == taught.strata
    assert interaction.reduction_over([0, 1, 2, 3]) == pytest.approx(interaction.reduction)


def test_an_equal_gain_from_teaching_on_both_sides_is_no_interaction() -> None:
    interaction = AnomalyInteraction(taught=paired(MIDDLE, FIRST), untaught=paired(MIDDLE, FIRST))

    assert interaction.reduction == pytest.approx(0.0)
    assert interaction.reduction_over([0, 1, 1, 3]) == pytest.approx(0.0)


def test_the_bootstrap_resamples_all_four_sides_under_one_draw() -> None:
    interaction = AnomalyInteraction(taught=paired(MIDDLE, FIRST), untaught=paired(LAST, LAST))

    difference = PairedUnitBootstrap(resamples=200, seed=3).compare(interaction)

    assert difference.reduction == pytest.approx(interaction.reduction)
    assert difference.relative_reduction is None
    assert difference.interval.low <= difference.reduction <= difference.interval.high


def test_pairs_over_other_units_or_other_events_are_refused() -> None:
    other_units = AnomalyRanking.of(
        prediction(unit, 0, event, score)
        for unit, event, score in (
            ("a", 0.0, 0.9),
            ("b", 1.0, 0.1),
            ("c", 0.0, 0.8),
            ("e", 2.0, 0.2),
        )
    )
    other_events = AnomalyRanking.of(
        prediction(unit, 0, event, score)
        for unit, event, score in (
            ("a", 1.0, 0.9),
            ("b", 0.0, 0.1),
            ("c", 0.0, 0.8),
            ("d", 2.0, 0.2),
        )
    )

    with pytest.raises(InvalidPairedAnomalyRankingsError, match="units"):
        AnomalyInteraction(
            taught=PairedAnomalyRankings(control=(other_units,), candidate=(other_units,)),
            untaught=paired(LAST, LAST),
        )
    with pytest.raises(InvalidPairedAnomalyRankingsError, match="windows or their events"):
        AnomalyInteraction(
            taught=PairedAnomalyRankings(control=(other_events,), candidate=(other_events,)),
            untaught=paired(LAST, LAST),
        )
