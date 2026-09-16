from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import (
    BackboneAlreadyDeliveredError,
    InvalidBackboneError,
)
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import TINY, WEIGHTS
from tests.support.handoff import (
    DELIVERED_AT,
    ORDERED_AT,
    RESULT_REF,
    backbone,
    pretraining_input,
)


def test_an_ordered_backbone_is_named_signed_and_not_ready() -> None:
    ordered = backbone()

    assert ordered.name == "test-experiment/first"
    assert not ordered.is_ready
    assert ordered.seed == 1
    assert ordered.parameter_count == TINY.parameter_count(ordered.input.vocabulary_size)


def test_delivering_the_weights_makes_a_new_ready_backbone_and_leaves_the_order() -> None:
    ordered = backbone()

    ready = ordered.deliver(RESULT_REF, WEIGHTS, DELIVERED_AT)

    assert ready.is_ready
    assert (ready.result, ready.artifact, ready.delivered_at) == (
        RESULT_REF,
        WEIGHTS,
        DELIVERED_AT,
    )
    assert ready.signature == ordered.signature
    assert not ordered.is_ready


def test_an_open_backbone_passes_and_a_ready_one_says_what_it_holds() -> None:
    ordered = backbone()

    ordered.require_open()
    with pytest.raises(BackboneAlreadyDeliveredError, match=f"already holds {WEIGHTS.key}"):
        ordered.deliver(RESULT_REF, WEIGHTS, DELIVERED_AT).require_open()


def test_a_backbone_that_is_ready_takes_no_second_delivery() -> None:
    ready = backbone().deliver(RESULT_REF, WEIGHTS, DELIVERED_AT)
    other = ArtifactRef("durable/other", Checksum.of_bytes(b"other weights"))

    with pytest.raises(BackboneAlreadyDeliveredError, match="already holds"):
        ready.deliver(RESULT_REF, other, DELIVERED_AT)


@pytest.mark.parametrize(
    "stated",
    [
        {"artifact": WEIGHTS},
        {"delivered_at": DELIVERED_AT},
        {"result": RESULT_REF},
        {"artifact": WEIGHTS, "delivered_at": DELIVERED_AT},
    ],
    ids=["artifact_alone", "time_alone", "result_alone", "without_the_result"],
)
def test_the_result_the_artifact_and_the_time_of_delivery_come_together(
    stated: dict[str, Any],
) -> None:
    with pytest.raises(InvalidBackboneError, match="come together"):
        backbone(**stated)


def test_a_delivery_before_the_order_is_refused() -> None:
    with pytest.raises(InvalidBackboneError, match="before it is ordered"):
        backbone(ordered_at=DELIVERED_AT).deliver(RESULT_REF, WEIGHTS, ORDERED_AT)


@pytest.mark.parametrize("field", ["run", "git_commit"])
@pytest.mark.parametrize("value", ["", " first", "first "])
def test_a_blank_or_padded_name_is_refused(field: str, value: str) -> None:
    with pytest.raises(InvalidBackboneError, match=field):
        backbone(**{field: value})


def test_the_parameter_count_follows_the_vocabulary_the_corpus_was_published_under() -> None:
    wider = backbone(input=pretraining_input(vocabulary_size=30))

    assert wider.parameter_count == TINY.parameter_count(30)
    assert wider.parameter_count > backbone().parameter_count
