from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import windows

CHECKSUM = Checksum.of_bytes(b"the windows a run reads")


def test_both_sides_are_named_and_kept_apart() -> None:
    stated = TrainingCorpus(
        name="control-a",
        checksum=CHECKSUM,
        training=windows(3, seed=1),
        validation=windows(2, seed=2),
        vocabulary_size=3,
    )

    assert (len(stated.training), len(stated.validation)) == (3, 2)


@pytest.mark.parametrize("side", ["training", "validation"])
def test_a_side_without_windows_is_refused(side: str) -> None:
    stated: dict[str, Any] = {
        "name": "control-a",
        "checksum": CHECKSUM,
        "training": windows(1, seed=1),
        "validation": windows(1, seed=2),
        "vocabulary_size": 3,
        side: [],
    }
    with pytest.raises(InvalidTrainingCorpusError, match=f"the {side} side"):
        TrainingCorpus(**stated)


@pytest.mark.parametrize(("field", "value"), [("name", " control "), ("vocabulary_size", 0)])
def test_a_corpus_no_run_could_read_is_refused(field: str, value: object) -> None:
    stated: dict[str, Any] = {
        "name": "control-a",
        "checksum": CHECKSUM,
        "training": windows(1, seed=1),
        "validation": windows(1, seed=2),
        "vocabulary_size": 3,
        field: value,
    }
    with pytest.raises(InvalidTrainingCorpusError):
        TrainingCorpus(**stated)
