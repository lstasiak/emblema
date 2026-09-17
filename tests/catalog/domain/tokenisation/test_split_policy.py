import pytest

from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.split_policy import NamedSplit, SeededSplit
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit

KEYS = [UnitKey(f"month-{number:02d}") for number in range(10)]


def test_a_drawn_split_is_the_draw_and_records_the_seed_it_was_drawn_with() -> None:
    policy = SeededSplit(0.3, seed=7)

    assert policy.applied_to(KEYS) == UnitSplit.by_seed(KEYS, 0.3, seed=7)
    assert policy.recorded_seed == 7


def test_a_named_split_holds_out_what_it_names_and_records_no_seed() -> None:
    policy = NamedSplit.of((KEYS[0], KEYS[4]))

    split = policy.applied_to(KEYS)

    assert split.validation == frozenset({KEYS[0], KEYS[4]})
    # Nothing was drawn, so there is no seed a reader could draw the same split with; the names
    # travel with the publication instead.
    assert policy.recorded_seed is None


def test_naming_the_units_puts_them_on_the_held_out_side_whatever_a_draw_would_do() -> None:
    """The point of naming: a draw at this fraction does not hold these units out."""
    named = NamedSplit.of((KEYS[0], KEYS[1])).applied_to(KEYS)
    drawn = SeededSplit(0.2, seed=1).applied_to(KEYS)

    assert named.validation == frozenset({KEYS[0], KEYS[1]})
    assert drawn.validation != named.validation


def test_a_policy_that_could_not_split_these_units_is_refused_where_it_is_applied() -> None:
    with pytest.raises(InvalidUnitSplitError, match="does not hold"):
        NamedSplit.of((UnitKey("absent"),)).applied_to(KEYS)
    with pytest.raises(InvalidUnitSplitError, match="fraction"):
        SeededSplit(1.5, seed=1).applied_to(KEYS)
