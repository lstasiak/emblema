import json

import pytest

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from tests.support.experiments import budget, configuration

CODEC = ExperimentConfigurationDocument()


def test_a_configuration_round_trips_through_its_document() -> None:
    stated = configuration(
        name="round-trip", precision=Precision.FP16, budget=budget(seed=9, warmup_epochs=1)
    )

    assert CODEC.decode(CODEC.encode(stated)) == stated


def test_the_reading_of_the_loss_travels_with_its_knee() -> None:
    bounded = configuration(loss=ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.5))

    document = CODEC.encode(bounded)

    assert document["objective"] == {"kind": "huber", "huber_delta": 1.5}
    assert CODEC.decode(document) == bounded


def test_the_document_is_plain_json_with_the_shape_stated_outright() -> None:
    document = CODEC.encode(configuration())

    assert json.loads(json.dumps(document)) == document
    assert document["architecture"] == {
        "width": 16,
        "heads": 2,
        "layers": 1,
        "feedforward_width": 32,
        "time_frequencies": 4,
    }


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ({"tier": "XL"}, "XL"),
        ({"precision": "fp8"}, "fp8"),
        ({"dropout": 1.5}, "dropout"),
        ({"corpus_fraction": 0}, "fraction"),
        ({"budget": {}}, "epochs"),
        ({"objective": {"kind": "mae", "huber_delta": 0.0}}, "mae"),
        ({"objective": {"kind": "mse", "huber_delta": 1.0}}, "no knee"),
        ({"architecture": None}, "architecture"),
    ],
)
def test_a_document_that_states_no_configuration_is_refused(
    broken: dict[str, object], message: str
) -> None:
    document = CODEC.encode(configuration()) | broken

    with pytest.raises(ValueError, match=message):
        CODEC.decode(document)
