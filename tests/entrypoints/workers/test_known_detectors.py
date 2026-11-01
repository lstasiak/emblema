"""The names the detectors compete under, and what each reads a window as.

A stored campaign refers to its candidates by these names, so they are asserted here rather
than left to whichever process happened to build the register last.
"""

from emblema.entrypoints.workers.known_detectors import KnownDetectors
from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
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
        recipe = arm.recipe(seed=1)
        if arm.features.asks_the_pretext:
            assert recipe.asked() == PretextAsking(backbone=WEIGHTS, draws=arm.draws)
        else:
            assert recipe.nearest() == arm.neighbours


def test_the_three_scored_by_an_error_ask_under_the_same_masks_and_knobs() -> None:
    pretext, linear, interpolation = (
        over()["pretext_error"],
        over()["linear_answer_error"],
        over()["interpolation_error"],
    )

    assert pretext.features is DetectorFeatures.PRETEXT_ERROR
    assert linear.features is DetectorFeatures.LINEAR_ANSWER_ERROR
    assert interpolation.features is DetectorFeatures.INTERPOLATION_ERROR
    for arm in (pretext, linear, interpolation):
        assert (arm.architecture, arm.backbone) == (None, WEIGHTS)
        assert (arm.draws, arm.contamination, arm.reference) == (
            PretextAsking.STANDARD_DRAWS,
            over()["frozen_knn"].contamination,
            over()["frozen_knn"].reference,
        )


def test_the_one_over_the_cells_reads_no_encoder_and_lays_them_out_as_the_states_are() -> None:
    cells = next(arm for arm in KnownDetectors.over(WEIGHTS) if arm.ref == KnownDetectors.CELLS)

    assert cells.features is DetectorFeatures.RAW_CELLS
    assert cells.architecture is None
    assert cells.backbone is None
    assert cells.recipe(seed=1).reading is None
    assert KnownDetectors.CELLS in KnownDetectors.refs()
