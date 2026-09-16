import json

import pytest

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.domain.training.precision import Precision
from tests.support.experiments import budget, configuration

CODEC = ExperimentConfigurationDocument()


def test_a_configuration_round_trips_through_its_document() -> None:
    stated = configuration(
        name="round-trip", precision=Precision.FP16, budget=budget(seed=9, warmup_epochs=1)
    )

    assert CODEC.decode(CODEC.encode(stated)) == stated


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
        ({"budget": {}}, "epochs"),
        ({"architecture": None}, "architecture"),
    ],
)
def test_a_document_that_states_no_configuration_is_refused(
    broken: dict[str, object], message: str
) -> None:
    document = CODEC.encode(configuration()) | broken

    with pytest.raises(ValueError, match=message):
        CODEC.decode(document)
