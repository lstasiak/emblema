from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from tests.support.experiments import configuration, corpus

READ = corpus(training=8, validation=4)


def test_the_shape_of_a_corpus_is_its_counts_beside_its_identity() -> None:
    assert READ.shape == TrainingCorpusShape(
        name=READ.name,
        checksum=READ.checksum,
        training_windows=8,
        validation_windows=4,
        vocabulary_size=READ.vocabulary_size,
    )


def test_a_run_signs_the_same_from_the_corpus_and_from_its_shape() -> None:
    stated = configuration()

    assert RunSignature.of_shape(stated, READ.shape) == RunSignature.of(stated, READ)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", " control "),
        ("training_windows", 0),
        ("validation_windows", 0),
        ("vocabulary_size", 0),
    ],
)
def test_a_shape_no_corpus_could_have_is_refused(field: str, value: Any) -> None:
    stated: dict[str, Any] = {
        "name": "control-a",
        "checksum": READ.checksum,
        "training_windows": 8,
        "validation_windows": 4,
        "vocabulary_size": 3,
        field: value,
    }
    with pytest.raises(InvalidTrainingCorpusError):
        TrainingCorpusShape(**stated)
