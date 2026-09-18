from dataclasses import replace

import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingMixtureError
from emblema.pretraining.domain.training.training_mixture_shape import TrainingMixtureShape
from tests.support.experiments import continued, corpus

FIRST = corpus().shape
SECOND = continued(name="second", seed=2).shape


def test_the_shape_names_and_sizes_the_mixture_by_its_corpora() -> None:
    shape = TrainingMixtureShape(corpora=(FIRST, SECOND))

    assert shape.name == "invented+second"
    assert [corpus.name for corpus in shape.corpora] == ["invented", "second"]
    assert shape.vocabulary_size == SECOND.vocabulary_size


def test_a_shape_without_a_corpus_is_refused() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="at least one"):
        TrainingMixtureShape(corpora=())


def test_a_corpus_named_twice_is_refused() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="twice"):
        TrainingMixtureShape(corpora=(FIRST, replace(FIRST, training_windows=6)))


def test_vocabularies_that_shrink_along_the_order_are_refused() -> None:
    with pytest.raises(InvalidTrainingMixtureError, match="do not shrink"):
        TrainingMixtureShape(corpora=(SECOND, FIRST))
