import pytest

from emblema.pretraining.domain.exceptions import InvalidCorpusPassesError
from emblema.pretraining.domain.training.corpus_passes import CorpusPasses


def test_a_corpus_is_read_as_many_times_as_stated() -> None:
    stated = CorpusPasses(corpus="physionet2012", passes=4)

    assert (stated.corpus, stated.passes) == ("physionet2012", 4)


@pytest.mark.parametrize("passes", [1, 0, -2])
def test_a_corpus_read_once_or_less_is_not_stated(passes: int) -> None:
    with pytest.raises(InvalidCorpusPassesError, match="at least 2"):
        CorpusPasses(corpus="physionet2012", passes=passes)


@pytest.mark.parametrize("corpus", ["", " skab", "skab "])
def test_a_corpus_without_a_clean_name_is_refused(corpus: str) -> None:
    with pytest.raises(InvalidCorpusPassesError, match="whitespace"):
        CorpusPasses(corpus=corpus, passes=2)
