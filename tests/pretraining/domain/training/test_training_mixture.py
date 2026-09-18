import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingMixtureError
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from tests.support.experiments import CHANNEL_NAMES, continued, corpus

FIRST = corpus()
SECOND = continued(name="second", seed=2, channels=2)
THIRD = corpus(name="third", seed=3, channels=(*SECOND.channels, "third/0", "third/1"))


def test_a_mixture_of_one_corpus_is_that_corpus() -> None:
    alone = TrainingMixture.of(FIRST)

    assert alone.name == FIRST.name
    assert alone.channels == FIRST.channels
    assert alone.vocabulary_size == FIRST.vocabulary_size
    assert alone.shape.corpora == (FIRST.shape,)


def test_a_mixture_is_named_and_sized_by_its_corpora_in_order() -> None:
    mixed = TrainingMixture.of(FIRST, SECOND, THIRD)

    assert mixed.name == "invented+second+third"
    assert [corpus.name for corpus in mixed.shape.corpora] == ["invented", "second", "third"]
    assert mixed.channels == THIRD.channels
    assert mixed.vocabulary_size == 7
    assert mixed.shape.vocabulary_size == 7


def test_a_corpus_left_out_of_the_chain_leaves_a_chain() -> None:
    mixed = TrainingMixture.of(FIRST, THIRD)

    assert mixed.vocabulary_size == THIRD.vocabulary_size


def test_two_corpora_published_apart_are_refused() -> None:
    """Both carry identifiers from one: a run over them would learn a collision silently."""
    apart = corpus(name="apart", seed=5, channels=("apart/x", "apart/y", "apart/z"))

    with pytest.raises(InvalidTrainingMixtureError, match="does not continue"):
        TrainingMixture.of(FIRST, apart)


def test_the_chain_is_read_in_the_order_given() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="do not shrink"):
        TrainingMixture.of(SECOND, FIRST)


def test_a_corpus_that_renames_an_earlier_channel_is_refused() -> None:
    renamed = corpus(name="renamed", seed=6, channels=(*CHANNEL_NAMES[:2], "renamed/c", "x"))

    with pytest.raises(InvalidTrainingMixtureError, match="does not continue"):
        TrainingMixture.of(FIRST, renamed)


def test_a_mixture_without_a_corpus_is_refused() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="at least one"):
        TrainingMixture.of()


def test_a_corpus_mixed_in_twice_is_refused() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="twice"):
        TrainingMixture.of(FIRST, corpus(seed=9))
