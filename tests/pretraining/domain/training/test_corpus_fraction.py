import math

import pytest

from emblema.pretraining.domain.exceptions import InvalidCorpusFractionError
from emblema.pretraining.domain.training.corpus_fraction import CorpusFraction


def test_a_corpus_is_read_at_the_fraction_stated() -> None:
    stated = CorpusFraction(corpus="utsd/Energy_london_smart_meters", fraction=0.25)

    assert (stated.corpus, stated.fraction) == ("utsd/Energy_london_smart_meters", 0.25)


def test_a_whole_corpus_may_be_stated_where_the_mixture_is_read_in_part() -> None:
    assert CorpusFraction(corpus="skab", fraction=1.0).fraction == 1.0


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5, math.nan, math.inf])
def test_a_fraction_no_run_could_read_is_refused(fraction: float) -> None:
    with pytest.raises(InvalidCorpusFractionError, match=r"\(0, 1\]"):
        CorpusFraction(corpus="skab", fraction=fraction)


@pytest.mark.parametrize("corpus", ["", " skab", "skab "])
def test_a_corpus_without_a_clean_name_is_refused(corpus: str) -> None:
    with pytest.raises(InvalidCorpusFractionError, match="whitespace"):
        CorpusFraction(corpus=corpus, fraction=0.5)
