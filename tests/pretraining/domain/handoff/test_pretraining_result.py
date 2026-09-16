from dataclasses import replace

import pytest

from emblema.pretraining.domain.exceptions import (
    InvalidPretrainingResultError,
    PretrainingResultRejectedError,
)
from emblema.pretraining.domain.training.run_signature import RunSignature
from tests.support.experiments import budget, configuration, corpus
from tests.support.handoff import (
    CHECKPOINT,
    CONFIGURATION,
    CORPUS,
    OTHER_COMMIT,
    backbone,
    backbone_id,
    result,
)


def test_the_signature_is_computed_from_what_the_result_states() -> None:
    assert result().signature == RunSignature.of(CONFIGURATION, CORPUS)
    assert result(corpus=corpus(seed=7).shape).signature != result().signature


def test_the_run_ordered_is_accepted_as_a_run_of_it() -> None:
    result().require_run_of(CONFIGURATION, CORPUS.shape, None)
    result(resumed_from=CHECKPOINT).require_run_of(CONFIGURATION, CORPUS.shape, CHECKPOINT)


def test_a_result_of_another_configuration_is_refused_naming_the_parameter() -> None:
    delivered = result(configuration=configuration(budget=budget(seed=2)))

    with pytest.raises(PretrainingResultRejectedError, match=r"seed=2 for 1"):
        delivered.require_run_of(CONFIGURATION, CORPUS.shape, None)


def test_a_result_over_other_data_is_refused_naming_the_checksums() -> None:
    other = corpus(seed=7)
    delivered = result(corpus=other.shape)

    with pytest.raises(PretrainingResultRejectedError) as refused:
        delivered.require_run_of(CONFIGURATION, CORPUS.shape, None)

    assert str(other.checksum) in str(refused.value)
    assert str(CORPUS.checksum) in str(refused.value)


def test_a_result_picked_up_from_another_checkpoint_is_refused() -> None:
    with pytest.raises(PretrainingResultRejectedError, match="picked up from the start"):
        result().require_run_of(CONFIGURATION, CORPUS.shape, CHECKPOINT)
    with pytest.raises(PretrainingResultRejectedError, match=CHECKPOINT.key):
        result(resumed_from=CHECKPOINT).require_run_of(CONFIGURATION, CORPUS.shape, None)


def test_the_delivery_a_backbone_waits_for_is_accepted() -> None:
    result().require_delivery_for(backbone())


def test_a_delivery_for_another_backbone_is_refused() -> None:
    with pytest.raises(PretrainingResultRejectedError, match="for backbone"):
        result(backbone=backbone_id(2)).require_delivery_for(backbone())


def test_a_delivery_made_with_other_code_is_refused() -> None:
    with pytest.raises(PretrainingResultRejectedError, match="commit"):
        result(git_commit=OTHER_COMMIT).require_delivery_for(backbone())


def test_a_delivery_of_another_configuration_is_refused_naming_the_parameter() -> None:
    delivered = result(configuration=configuration(budget=budget(seed=2)))

    with pytest.raises(PretrainingResultRejectedError, match=r"seed=2 for 1"):
        delivered.require_delivery_for(backbone())


def test_a_delivery_over_other_data_is_refused_naming_the_checksums() -> None:
    other = corpus(seed=7)
    delivered = result(corpus=other.shape)

    with pytest.raises(PretrainingResultRejectedError, match="ordered on") as refused:
        delivered.require_delivery_for(backbone())

    assert str(other.checksum) in str(refused.value)
    assert str(CORPUS.checksum) in str(refused.value)


def test_a_delivery_of_the_same_data_read_as_other_windows_is_refused_by_signature() -> None:
    # The same block and vocabulary, split into other counts: nothing but the signature, which
    # the registry keeps and the windows are not needed for, can tell the two runs apart.
    delivered = result(corpus=replace(CORPUS.shape, training_windows=6))

    with pytest.raises(PretrainingResultRejectedError, match=r"signs run.*6 training"):
        delivered.require_delivery_for(backbone())


def test_a_blank_commit_is_refused() -> None:
    with pytest.raises(InvalidPretrainingResultError, match="git_commit"):
        result(git_commit="")
