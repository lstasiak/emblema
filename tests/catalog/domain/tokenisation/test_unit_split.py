import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit

KEYS = [UnitKey(f"engine-{number:02d}") for number in range(10)]
KEY_SETS = st.sets(
    st.integers(min_value=0, max_value=999).map(lambda n: UnitKey(f"u{n}")), min_size=2, max_size=40
)


def test_the_same_units_fraction_and_seed_give_the_same_split() -> None:
    assert UnitSplit.by_seed(KEYS, 0.3, seed=7) == UnitSplit.by_seed(KEYS, 0.3, seed=7)


def test_the_order_units_arrive_in_does_not_matter() -> None:
    assert UnitSplit.by_seed(reversed(KEYS), 0.3, seed=7) == UnitSplit.by_seed(KEYS, 0.3, seed=7)


def test_a_corpus_that_grew_keeps_the_units_it_already_had_where_they_were() -> None:
    grown = [*KEYS, UnitKey("engine-10")]

    before = UnitSplit.by_seed(KEYS, 0.3, seed=7)
    after = UnitSplit.by_seed(grown, 0.3, seed=7)

    moved = [key for key in KEYS if (key in before.validation) != (key in after.validation)]
    assert len(moved) <= 1  # one more unit displaces at most the one at the cut


def test_the_sides_are_the_digest_of_the_keys_not_the_interpreters_shuffle() -> None:
    # Pinned: a split travels with the artefacts fitted under it, so the same seed must give the
    # same sides on every Python this runs on.
    split = UnitSplit.by_seed(KEYS, 0.3, seed=7)

    assert sorted(str(key) for key in split.validation) == ["engine-02", "engine-04", "engine-06"]


def test_another_seed_gives_another_split() -> None:
    assert UnitSplit.by_seed(KEYS, 0.3, seed=7) != UnitSplit.by_seed(KEYS, 0.3, seed=8)


def test_the_validation_side_takes_the_whole_units_that_fit_the_fraction() -> None:
    split = UnitSplit.by_seed(KEYS, 0.25, seed=1)

    assert (len(split.training), len(split.validation)) == (8, 2)


@given(keys=KEY_SETS, fraction=st.floats(min_value=0.05, max_value=0.95), seed=st.integers())
def test_every_unit_lands_on_exactly_one_side(
    keys: set[UnitKey], fraction: float, seed: int
) -> None:
    if int(len(keys) * fraction) == 0:
        return
    split = UnitSplit.by_seed(keys, fraction, seed)

    assert split.training | split.validation == keys
    assert not split.training & split.validation


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_the_fraction_lies_strictly_between_zero_and_one(fraction: float) -> None:
    with pytest.raises(InvalidUnitSplitError, match="strictly between"):
        UnitSplit.by_seed(KEYS, fraction, seed=0)


def test_too_few_units_for_the_fraction_is_an_error_not_an_empty_side() -> None:
    with pytest.raises(InvalidUnitSplitError, match="no unit for validation"):
        UnitSplit.by_seed(KEYS[:2], 0.2, seed=0)


def test_unit_keys_must_be_unique() -> None:
    with pytest.raises(InvalidUnitSplitError, match="unique"):
        UnitSplit.by_seed([*KEYS, KEYS[0]], 0.5, seed=0)


def test_both_sides_hold_a_unit() -> None:
    with pytest.raises(InvalidUnitSplitError, match="at least one unit"):
        UnitSplit(frozenset(KEYS), frozenset())


def test_the_sides_share_no_unit() -> None:
    with pytest.raises(InvalidUnitSplitError, match="both sides"):
        UnitSplit(frozenset(KEYS[:3]), frozenset(KEYS[2:]))


def test_a_split_holds_out_exactly_the_units_named() -> None:
    split = UnitSplit.of_held_out(KEYS, (KEYS[2], KEYS[0]))

    assert split.validation == frozenset({KEYS[0], KEYS[2]})
    assert split.training == frozenset(KEYS) - split.validation


def test_naming_every_unit_or_none_of_them_leaves_a_side_empty() -> None:
    with pytest.raises(InvalidUnitSplitError, match="at least one unit"):
        UnitSplit.of_held_out(KEYS, KEYS)
    with pytest.raises(InvalidUnitSplitError, match="at least one unit"):
        UnitSplit.of_held_out(KEYS, ())


def test_a_unit_the_corpus_does_not_hold_cannot_be_held_out() -> None:
    with pytest.raises(InvalidUnitSplitError, match="does not hold"):
        UnitSplit.of_held_out(KEYS, (UnitKey("absent"),))


def test_neither_side_of_a_named_split_takes_a_unit_twice() -> None:
    with pytest.raises(InvalidUnitSplitError, match="held-out keys"):
        UnitSplit.of_held_out(KEYS, (KEYS[0], KEYS[0]))
    with pytest.raises(InvalidUnitSplitError, match="unit keys"):
        UnitSplit.of_held_out((*KEYS, KEYS[0]), (KEYS[1],))
