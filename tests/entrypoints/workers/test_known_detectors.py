"""The names the detectors compete under, and what each reads a window as.

A stored campaign refers to its candidates by these names, so they are asserted here rather
than left to whichever process happened to build the register last.
"""

from emblema.entrypoints.workers.known_detectors import KnownDetectors
from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.layer_reading import LayerReading
from tests.evaluation.support import WEIGHTS


def over() -> dict[str, DetectorArm]:
    return {str(arm.ref): arm for arm in KnownDetectors.over(WEIGHTS)}


def test_every_detector_is_registered_under_the_name_it_is_competed_by() -> None:
    assert list(over()) == [str(ref) for ref in KnownDetectors.refs()]


def test_the_two_over_an_encoder_share_its_shape_and_differ_in_its_weights_alone() -> None:
    frozen, untrained = over()["frozen_knn"], over()["untrained_knn"]

    assert (frozen.architecture, frozen.backbone) == (WEIGHTS, WEIGHTS)
    assert (untrained.architecture, untrained.backbone) == (WEIGHTS, None)
    assert (frozen.encoder, frozen.pooling) == (untrained.encoder, untrained.pooling)
    assert frozen.encoder.layer == LayerReading.of("concat")
    assert frozen.pooling == HeadPooling.mean()
    assert (frozen.neighbours, frozen.contamination) == (
        untrained.neighbours,
        untrained.contamination,
    )


def test_the_one_over_the_summaries_reads_no_encoder_under_the_same_knobs() -> None:
    summaries, frozen = over()["summaries_knn"], over()["frozen_knn"]

    assert summaries.features is DetectorFeatures.SUMMARIES
    assert (summaries.architecture, summaries.backbone) == (None, None)
    assert (summaries.neighbours, summaries.contamination) == (
        frozen.neighbours,
        frozen.contamination,
    )


def test_the_one_over_the_counts_reads_the_sampling_alone_under_the_same_knobs() -> None:
    counts, frozen = over()["counts_knn"], over()["frozen_knn"]

    assert counts.features is DetectorFeatures.READING_COUNTS
    assert (counts.architecture, counts.backbone) == (None, None)
    assert (counts.neighbours, counts.contamination, counts.reference) == (
        frozen.neighbours,
        frozen.contamination,
        frozen.reference,
    )


def test_every_detector_stands_up_as_a_recipe() -> None:
    for arm in over().values():
        assert arm.recipe(seed=1).neighbours == arm.neighbours
