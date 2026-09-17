import json

import pytest

from emblema.pretraining.adapters.documents.pretraining_result_json import PretrainingResultJson
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from tests.support.experiments import WEIGHTS, epoch_outcome, validated
from tests.support.handoff import CHECKPOINT, result

CODEC = PretrainingResultJson()


def test_a_result_round_trips_with_a_resume_and_every_checkpoint() -> None:
    epochs = (
        epoch_outcome(3, checkpoint=CHECKPOINT),
        epoch_outcome(4),
        epoch_outcome(5, checkpoint=CHECKPOINT, backbone=WEIGHTS),
    )
    reported = result(
        resumed_from=CHECKPOINT, outcome=TrainingOutcome(backbone=WEIGHTS, epochs=epochs)
    )

    assert CODEC.decode(CODEC.encode(reported)) == reported


def test_a_result_of_a_run_from_the_start_round_trips() -> None:
    reported = result()

    decoded = CODEC.decode(CODEC.encode(reported))

    assert decoded == reported
    assert decoded.resumed_from is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "not JSON"),
        (b'"a string"', "not an object"),
        (b'{"format": "emblema.pretraining-order", "version": 1}', "not a result"),
    ],
)
def test_bytes_that_are_not_a_result_are_refused(content: bytes, message: str) -> None:
    with pytest.raises(UnreadableHandoffDocumentError, match=message):
        CODEC.decode(content)


def test_a_result_whose_outcome_breaks_the_domain_rules_is_refused() -> None:
    document = json.loads(CODEC.encode(result()))
    document["outcome"]["epochs"] = document["outcome"]["epochs"][::-1]

    with pytest.raises(UnreadableHandoffDocumentError, match="consecutive"):
        CODEC.decode(json.dumps(document).encode())


def test_a_result_with_a_field_missing_names_it() -> None:
    document = json.loads(CODEC.encode(result()))
    del document["corpus"]["vocabulary_size"]

    with pytest.raises(UnreadableHandoffDocumentError, match="vocabulary_size"):
        CODEC.decode(json.dumps(document).encode())


def test_every_corpus_an_epoch_validated_travels_in_the_result() -> None:
    mixed = epoch_outcome(
        0,
        backbone=WEIGHTS,
        validation=(
            validated(corpus="cmapss", tokens=10, loss=0.5, trivial=1.0),
            validated(corpus="esa_ad", tokens=90, loss=6.0, trivial=5.0),
        ),
    )
    stated = result(outcome=TrainingOutcome(backbone=WEIGHTS, epochs=(mixed,)))

    read = CODEC.decode(CODEC.encode(stated))

    scored = read.outcome.epochs[0].validation
    assert [(entry.corpus, entry.tokens) for entry in scored] == [("cmapss", 10), ("esa_ad", 90)]
    assert read.outcome.epochs[0].validation_loss == mixed.validation_loss
    assert read.outcome.epochs[0].relative_validation == mixed.relative_validation
