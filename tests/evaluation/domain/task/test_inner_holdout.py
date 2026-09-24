import pytest

from emblema.evaluation.domain.exceptions import InvalidInnerHoldoutError
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from tests.evaluation.support import units

TWENTY = units(*(f"u{index:02d}" for index in range(20)))


def test_one_in_every_so_many_units_is_held_out_and_the_rest_learnt_from() -> None:
    fitted, held = InnerHoldout(one_in=5).divided(TWENTY, seed=1)

    assert len(held) == 4
    assert fitted | held == TWENTY
    assert not fitted & held


def test_the_same_seed_divides_the_same_way_and_another_seed_otherwise() -> None:
    holdout = InnerHoldout(one_in=5)

    assert holdout.divided(TWENTY, seed=1) == holdout.divided(TWENTY, seed=1)
    assert holdout.divided(TWENTY, seed=1) != holdout.divided(TWENTY, seed=2)


def test_the_share_scored_per_unit_learnt_is_what_the_rule_corrects_by() -> None:
    assert InnerHoldout(one_in=5).test_to_train == pytest.approx(0.25)


def test_one_in_fewer_than_two_is_refused() -> None:
    with pytest.raises(InvalidInnerHoldoutError):
        InnerHoldout(one_in=1)


def test_too_few_units_to_leave_one_on_each_side_are_refused() -> None:
    with pytest.raises(InvalidInnerHoldoutError, match="leave none"):
        InnerHoldout(one_in=2).divided(units("a"), seed=1)
