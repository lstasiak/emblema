from math import ceil

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.pretraining.domain.exceptions import InvalidCorpusShareError
from emblema.pretraining.domain.training.corpus_share import CorpusShare

UNITS = tuple(f"u{number:02d}" for number in range(1, 11))


def test_the_whole_corpus_is_every_unit_in_the_order_given() -> None:
    assert CorpusShare(fraction=1.0, seed=1).select(UNITS) == UNITS
    assert CorpusShare(fraction=1.0, seed=1).is_whole


def test_a_share_takes_the_units_that_fit_the_fraction_rounded_up() -> None:
    assert len(CorpusShare(fraction=0.25, seed=1).select(UNITS)) == 3
    assert len(CorpusShare(fraction=0.1, seed=1).select(("a", "b", "c"))) == 1
    # Three tenths of ten is three: the binary 0.3 × 10 must not round up to four.
    assert len(CorpusShare(fraction=0.3, seed=1).select(UNITS)) == 3


def test_a_share_keeps_the_order_the_units_were_given_in() -> None:
    chosen = CorpusShare(fraction=0.5, seed=1).select(UNITS)

    assert list(chosen) == [unit for unit in UNITS if unit in chosen]


def test_the_shares_of_one_seed_are_nested() -> None:
    tenth = set(CorpusShare(fraction=0.1, seed=1).select(UNITS))
    quarter = set(CorpusShare(fraction=0.25, seed=1).select(UNITS))
    half = set(CorpusShare(fraction=0.5, seed=1).select(UNITS))

    assert tenth <= quarter <= half


def test_another_seed_ranks_the_units_differently() -> None:
    assert CorpusShare(fraction=0.3, seed=1).select(UNITS) != CorpusShare(
        fraction=0.3, seed=2
    ).select(UNITS)


def test_the_ranking_is_pinned() -> None:
    # The digest is framed apart from the Catalog's split, so a run seeded like its corpus's
    # split does not read the units just past the validation cut. Changing the framing changes
    # every share read so far, and is done here on purpose or not at all.
    assert CorpusShare(fraction=0.3, seed=1).select(UNITS) == ("u04", "u07", "u08")


@given(
    st.lists(st.text(min_size=1, max_size=4), min_size=1, max_size=12, unique=True),
    st.floats(min_value=0.01, max_value=1.0),
    st.integers(min_value=0, max_value=1000),
)
def test_a_share_is_a_subset_of_the_right_size_whatever_the_units(
    units: list[str], fraction: float, seed: int
) -> None:
    chosen = CorpusShare(fraction=fraction, seed=seed).select(units)

    assert set(chosen) <= set(units)
    assert 1 <= len(chosen) <= len(units)
    assert len(chosen) == ceil(round(fraction * len(units), 9))


def test_a_unit_that_repeats_is_refused() -> None:
    with pytest.raises(InvalidCorpusShareError, match="unique"):
        CorpusShare(fraction=0.5, seed=1).select(("a", "a"))


@pytest.mark.parametrize("fraction", [0.0, 1.0001, -1.0, float("inf"), float("nan")])
def test_a_fraction_no_run_could_read_is_refused(fraction: float) -> None:
    with pytest.raises(InvalidCorpusShareError):
        CorpusShare(fraction=fraction, seed=1)
