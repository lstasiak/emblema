from dataclasses import replace
from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import (
    BackboneAlreadyDeliveredError,
    IncompatibleStartError,
    InvalidBackboneError,
)
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import TINY, WEIGHTS, configuration
from tests.support.handoff import (
    DELIVERED_AT,
    ORDERED_AT,
    RESULT_REF,
    backbone,
    backbone_id,
    pretraining_input,
)


def test_an_ordered_backbone_is_named_signed_and_not_ready() -> None:
    ordered = backbone()

    assert ordered.name == "test-experiment/first"
    assert not ordered.is_ready
    assert ordered.seed == 1
    assert ordered.parameter_count == TINY.parameter_count(ordered.vocabulary_size)


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
    wider = backbone(inputs=(pretraining_input(vocabulary_size=30),))

    assert wider.parameter_count == TINY.parameter_count(30)
    assert wider.parameter_count > backbone().parameter_count


def test_a_backbone_over_several_corpora_trains_under_the_last_one_s_vocabulary() -> None:
    mixed = backbone(
        inputs=(
            pretraining_input(),
            pretraining_input(corpus="second", vocabulary_size=5),
            pretraining_input(corpus="third", vocabulary_size=9),
        )
    )

    assert mixed.vocabulary_size == 9
    assert mixed.parameter_count == TINY.parameter_count(9)


def test_a_backbone_over_no_corpus_is_refused() -> None:
    with pytest.raises(InvalidBackboneError, match="at least one corpus"):
        backbone(inputs=())


def test_a_corpus_read_twice_is_refused() -> None:
    with pytest.raises(InvalidBackboneError, match="read twice"):
        backbone(inputs=(pretraining_input(), pretraining_input(vocabulary_size=5)))


def test_vocabularies_that_shrink_along_the_corpora_are_refused() -> None:
    with pytest.raises(InvalidBackboneError, match="do not shrink"):
        backbone(
            inputs=(pretraining_input(), pretraining_input(corpus="second", vocabulary_size=2))
        )


CONTINUED = configuration(name="continued", start=WEIGHTS, trained=TrainedParameters.CHANNEL_ROWS)


def test_a_backbone_continued_from_stored_weights_names_its_parent_and_one_from_the_seed_none() -> (
    None
):
    child = backbone(id=backbone_id(2), configuration=CONTINUED, parent=backbone_id(1))

    assert child.parent == backbone_id(1)
    assert backbone().parent is None
    with pytest.raises(InvalidBackboneError, match="names its parent"):
        backbone(configuration=CONTINUED)
    with pytest.raises(InvalidBackboneError, match="names none"):
        backbone(parent=backbone_id(1))


def test_a_ready_backbone_of_the_run_s_shape_and_objective_can_be_continued() -> None:
    ready = backbone().deliver(RESULT_REF, WEIGHTS, DELIVERED_AT)

    ready.require_continuable_by(CONTINUED)


def test_a_backbone_without_weights_cannot_be_continued() -> None:
    with pytest.raises(IncompatibleStartError, match="holds no weights"):
        backbone().require_continuable_by(CONTINUED)


@pytest.mark.parametrize(
    ("differing", "label"),
    [
        ({"decoder_layers": 2}, "decoder"),
        ({"target": ObjectiveTarget.latent(layers=1)}, "target"),
        ({"architecture": replace(TINY, width=32, feedforward_width=64)}, "shape"),
    ],
    ids=["decoder", "target", "shape"],
)
def test_a_run_of_another_shape_objective_or_decoder_cannot_continue_the_weights(
    differing: dict[str, object], label: str
) -> None:
    ready = backbone().deliver(RESULT_REF, WEIGHTS, DELIVERED_AT)

    with pytest.raises(IncompatibleStartError, match=f"another {label}"):
        ready.require_continuable_by(configuration(start=WEIGHTS, **differing))
