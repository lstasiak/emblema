import pytest

from emblema.evaluation.adapters.synthetic.known_answer import KnownAnswer
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors


def test_the_pairs_scatter_around_the_levels_stated_and_the_reduction_is_known() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0, units=40)

    paired = known.paired(seed=1)

    assert known.true_reduction == 5.0
    assert paired.rmse_control == pytest.approx(30.0, rel=0.1)
    assert paired.rmse_candidate == pytest.approx(25.0, rel=0.1)
    assert len(paired.units) == 40


def test_every_repeat_shares_the_units_and_their_difficulty_and_has_its_own_luck() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=30.0, units=6, repeat_scatter=0.2)

    first, second = known.repeats(2, seed=4)

    assert first.units == second.units
    assert first != second
    # One difficulty per unit, shared by both sides: without luck the two sides are the same.
    steady = KnownAnswer(control_rmse=30.0, candidate_rmse=30.0, units=6, repeat_scatter=0.0)
    alike = steady.paired(seed=4)
    assert [e.squared_error for e in alike.control] == [e.squared_error for e in alike.candidate]


def test_the_same_seed_draws_the_same_pairs() -> None:
    known = KnownAnswer(control_rmse=30.0, candidate_rmse=25.0)

    assert known.repeats(3, seed=9) == known.repeats(3, seed=9)
    assert isinstance(known.paired(seed=9), PairedUnitErrors)
