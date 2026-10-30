"""What the detectors are called and set to, read out of their names alone."""

import pytest

from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.adapters.candidates.detector_catalogue import DetectorCatalogue
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    InvalidDetectorRecipeError,
    UnknownCandidateError,
    UnknownKnobError,
)
from emblema.evaluation.domain.transfer.layer_reading import LayerReading
from tests.evaluation.support import WEIGHTS

FROZEN = DetectorArm(
    ref=CandidateRef("frozen_knn"),
    features=DetectorFeatures.ENCODER_STATES,
    architecture=WEIGHTS,
    backbone=WEIGHTS,
)
UNTRAINED = DetectorArm(
    ref=CandidateRef("untrained_knn"),
    features=DetectorFeatures.ENCODER_STATES,
    architecture=WEIGHTS,
    backbone=None,
)
SUMMARIES = DetectorArm(
    ref=CandidateRef("summaries_knn"),
    features=DetectorFeatures.SUMMARIES,
    architecture=None,
    backbone=None,
)
CATALOGUE = DetectorCatalogue((FROZEN, UNTRAINED, SUMMARIES))


def test_a_detector_declares_no_compute_budget_and_names_the_weights_it_reads() -> None:
    frozen = CATALOGUE.describe(CandidateRef("frozen_knn"))
    summaries = CATALOGUE.describe(CandidateRef("summaries_knn"))

    assert (frozen.kind, frozen.budget, frozen.starts_from) == (
        CandidateKind.DETECTOR,
        None,
        WEIGHTS,
    )
    assert CATALOGUE.describe(CandidateRef("untrained_knn")).starts_from is None
    assert summaries.starts_from is None
    assert {p.name for p in summaries.method.parameters} == {
        "features",
        "neighbours",
        "contamination",
    }
    assert {p.name for p in frozen.method.parameters} >= {"architecture", "layer", "pooling"}


def test_a_variant_turns_the_detectors_knobs_and_the_encoders_reading() -> None:
    turned = CATALOGUE.arm_of(
        CandidateRef("frozen_knn@contamination=0.05,layer=concat,neighbours=10")
    )

    assert (turned.neighbours, turned.contamination) == (10, 0.05)
    assert turned.encoder.layer == LayerReading.of("concat")
    assert turned.ref == CandidateRef("frozen_knn@contamination=0.05,layer=concat,neighbours=10")


def test_the_description_of_a_variant_records_the_knobs_turned() -> None:
    described = CATALOGUE.describe(CandidateRef("summaries_knn@neighbours=3"))

    assert described.ref == CandidateRef("summaries_knn@neighbours=3")
    assert {p.name: p.value for p in described.method.parameters}["neighbours"] == "3"


@pytest.mark.parametrize(
    "name",
    [
        "frozen_knn@neighbours=0",
        "frozen_knn@contamination=1",
        "frozen_knn@neighbours=five",
        "summaries_knn@layer=concat",
        "frozen_knn@pooling=attention",
        "frozen_knn@dropout=0.1",
        "frozen_knn@neighbours=5",
        "trees_knn",
        "frozen_knn@epochs=3",
        "frozen_knn@reference=month",
        "frozen_knn@reference=fitting",
    ],
    ids=[
        "no neighbour",
        "all contamination",
        "not a count",
        "a layer for the summaries",
        "a pooling that learns",
        "a dropout",
        "the default under another name",
        "unknown base",
        "unknown knob",
        "an unknown reference",
        "the default reference under another name",
    ],
)
def test_a_name_that_is_no_detector_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        CATALOGUE.arm_of(CandidateRef(name))


def test_an_arm_over_the_summaries_turns_no_knob_of_an_encoder() -> None:
    with pytest.raises(UnknownKnobError, match="nothing else"):
        SUMMARIES.tuned("pooling", "tail")


def test_an_arm_whose_features_and_encoder_disagree_is_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="names an encoder"):
        DetectorArm(
            ref=CandidateRef("x"),
            features=DetectorFeatures.SUMMARIES,
            architecture=WEIGHTS,
            backbone=None,
        )
    with pytest.raises(InvalidDetectorRecipeError, match="names no encoder"):
        DetectorArm(
            ref=CandidateRef("x"),
            features=DetectorFeatures.ENCODER_STATES,
            architecture=None,
            backbone=None,
        )
    with pytest.raises(InvalidDetectorRecipeError, match="no shape"):
        DetectorArm(
            ref=CandidateRef("x"),
            features=DetectorFeatures.SUMMARIES,
            architecture=None,
            backbone=WEIGHTS,
        )


def test_the_recipe_of_an_arm_carries_the_cells_seed_and_the_encoders_reading() -> None:
    recipe = FROZEN.recipe(seed=4)

    assert recipe.seed == 4
    assert recipe.reading is not None
    assert (recipe.reading.architecture, recipe.reading.backbone) == (WEIGHTS, WEIGHTS)
    assert SUMMARIES.recipe(seed=1).reading is None


def test_a_detector_read_against_its_unit_says_so_and_one_against_the_fit_says_nothing() -> None:
    turned = CATALOGUE.arm_of(CandidateRef("summaries_knn@reference=unit"))
    described = CATALOGUE.describe(CandidateRef("summaries_knn@reference=unit"))

    assert turned.recipe(seed=1).reference is ScoreReference.OWN_UNIT
    assert {p.name: p.value for p in described.method.parameters}["reference"] == "unit"
    # A campaign declared before the reference was a knob keeps the design it was stored with.
    assert "reference" not in {
        p.name for p in CATALOGUE.describe(CandidateRef("summaries_knn")).method.parameters
    }
    assert FROZEN.recipe(seed=1).reference is ScoreReference.FITTING_SIDE


PRETEXT = DetectorArm(
    ref=CandidateRef("pretext_error"),
    features=DetectorFeatures.PRETEXT_ERROR,
    architecture=None,
    backbone=WEIGHTS,
)
ASKING = DetectorCatalogue((PRETEXT, SUMMARIES))


def test_a_detector_scored_by_an_error_names_its_draws_and_masks_and_no_neighbours() -> None:
    described = ASKING.describe(CandidateRef("pretext_error@draws=4"))
    stated = {p.name: p.value for p in described.method.parameters}

    assert stated["draws"] == "4"
    assert stated["masks"] == WEIGHTS.key
    assert "neighbours" not in stated
    assert ASKING.arm_of(CandidateRef("pretext_error@draws=4")).recipe(seed=1).asked().draws == 4


@pytest.mark.parametrize(
    "name",
    ["pretext_error@neighbours=3", "summaries_knn@draws=3", "pretext_error@draws=0"],
    ids=["neighbours of an error", "draws of a distance", "no draw"],
)
def test_a_knob_the_score_does_not_have_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        ASKING.arm_of(CandidateRef(name))


def test_an_arm_that_asks_the_pretext_names_a_backbone_and_no_shape() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="no backbone to ask"):
        DetectorArm(
            ref=CandidateRef("x"),
            features=DetectorFeatures.PRETEXT_ERROR,
            architecture=None,
            backbone=None,
        )
    with pytest.raises(InvalidDetectorRecipeError, match="names an encoder"):
        DetectorArm(
            ref=CandidateRef("x"),
            features=DetectorFeatures.LINEAR_ANSWER_ERROR,
            architecture=WEIGHTS,
            backbone=WEIGHTS,
        )
