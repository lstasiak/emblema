import pytest

from emblema.pretraining.domain.exceptions import InvalidCorpusValidationError
from emblema.pretraining.domain.training.corpus_validation import CorpusValidation
from tests.support.experiments import validated


def test_a_corpus_is_read_against_what_nothing_learnt_costs_on_it() -> None:
    scored = validated(corpus="esa_ad", tokens=1_379_092, loss=6.35, trivial=5.27)

    assert scored.relative == pytest.approx(6.35 / 5.27)
    # Under one the corpus was learnt; at one the trivial predictor was matched and no more.
    assert validated(loss=0.5, trivial=1.0).relative == 0.5
    assert validated(loss=1.0, trivial=1.0).relative == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus", ""),
        ("corpus", " esa_ad"),
        ("tokens", 0),
        ("tokens", -1),
        ("loss", -0.1),
        ("loss", float("nan")),
        ("trivial", 0.0),
        ("trivial", -1.0),
        ("trivial", float("inf")),
    ],
)
def test_a_validation_that_says_nothing_about_a_corpus_is_refused(
    field: str, value: object
) -> None:
    with pytest.raises(InvalidCorpusValidationError):
        validated(**{field: value})


def test_a_corpus_the_trivial_predictor_gets_right_holds_nothing_to_read() -> None:
    """A relative loss over a side of exact zeros is not a number, so the side is refused."""
    with pytest.raises(InvalidCorpusValidationError, match="positive"):
        CorpusValidation(corpus="flat", tokens=10, loss=0.0, trivial=0.0)
