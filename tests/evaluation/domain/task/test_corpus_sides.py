import pytest

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from tests.evaluation.support import sides, units


def test_a_corpus_reports_the_units_it_fitted_on_and_the_ones_it_held_out() -> None:
    published = sides(units("a", "b"), units("c"))

    assert published.training == units("a", "b")
    assert published.validation == units("c")


@pytest.mark.parametrize("name", ["", " turbofans"])
def test_a_corpus_whose_name_is_blank_or_padded_is_refused(name: str) -> None:
    with pytest.raises(InvalidTaskSplitError, match="corpus name"):
        CorpusSides(corpus=name, training=units("a"), validation=units("c"))


def test_a_corpus_that_held_nothing_out_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="must hold a unit"):
        CorpusSides(corpus="turbofans", training=units("a"), validation=units())


def test_a_unit_on_both_sides_of_the_corpus_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="both sides"):
        CorpusSides(corpus="turbofans", training=units("a", "c"), validation=units("c"))
