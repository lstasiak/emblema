import numpy as np
import pytest

from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError

FITTING = np.arange(100, dtype=np.float64)


def test_the_level_leaves_the_contamination_of_the_fitting_scores_above_it() -> None:
    alarm = ScoreAlarm.set_over(FITTING, contamination=0.02)

    above = alarm.above(FITTING)

    assert alarm.level == pytest.approx(np.quantile(FITTING, 0.98))
    assert int((above > 0).sum()) == 2


def test_a_score_read_against_its_unit_is_its_place_among_the_unit_s_own() -> None:
    units = ["a"] * 50 + ["b"] * 50
    shifted = FITTING + np.where(np.arange(100) >= 50, 1000.0, 0.0)
    alarm = ScoreAlarm.set_over(shifted, contamination=0.02, units=units)

    read = alarm.above(shifted, units=units) + alarm.level

    # The jump between the units is gone: each unit sits around its own median.
    assert np.median(read[:50]) == pytest.approx(0.0)
    assert np.median(read[50:]) == pytest.approx(0.0)
    assert alarm.per_unit


def test_a_unit_without_a_spread_borrows_the_fitting_side_s() -> None:
    units = ["a"] * 50 + ["b"] * 50
    alarm = ScoreAlarm.set_over(FITTING, contamination=0.02, units=units)

    # Two scores alike and a third: the unit's median absolute deviation is zero.
    read = alarm.above(np.array([5.0, 5.0, 9.0]), units=["c", "c", "c"]) + alarm.level

    assert alarm.unit_spread == pytest.approx(1.4826 * 25.0)
    assert read == pytest.approx([0.0, 0.0, 4.0 / alarm.unit_spread])


def test_scores_with_units_the_alarm_does_not_read_are_refused_and_the_other_way_round() -> None:
    plain = ScoreAlarm.set_over(FITTING, contamination=0.02)
    per_unit = ScoreAlarm.set_over(FITTING, contamination=0.02, units=["a", "b"] * 50)

    with pytest.raises(ValueError, match="reads no unit"):
        plain.above(FITTING, units=["a"] * 100)
    with pytest.raises(ValueError, match="was given no units"):
        per_unit.require(None)


def test_units_none_of_which_holds_two_scores_are_refused() -> None:
    with pytest.raises(InvalidScoredOutcomeError, match="no unit holds two"):
        ScoreAlarm.set_over(FITTING[:3], contamination=0.02, units=["a", "b", "c"])


def test_units_that_do_not_name_one_unit_per_score_are_refused() -> None:
    with pytest.raises(ValueError, match="2 units were given for 3 rows"):
        ScoreAlarm.set_over(FITTING[:3], contamination=0.02, units=["a", "a"])
