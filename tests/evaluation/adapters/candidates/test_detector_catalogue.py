"""What the detectors are called and set to, read out of their names alone."""

import pytest

from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.adapters.candidates.detector_catalogue import DetectorCatalogue
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.detection.taught_head import TaughtHead
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


def test_a_detector_by_the_whitened_distance_says_so_and_names_no_neighbours() -> None:
    turned = CATALOGUE.arm_of(CandidateRef("frozen_knn@distance=whitened"))
    described = CATALOGUE.describe(CandidateRef("frozen_knn@distance=whitened"))
    stated = {p.name: p.value for p in described.method.parameters}

    assert turned.recipe(seed=1).distance is ScoreDistance.WHITENED
    assert turned.recipe(seed=1).neighbours is None
    assert stated["distance"] == "whitened"
    assert "neighbours" not in stated
    # A campaign declared before the distance was a knob keeps the design it was stored with.
    assert "distance" not in {
        p.name for p in CATALOGUE.describe(CandidateRef("frozen_knn")).method.parameters
    }
    assert (
        CATALOGUE.arm_of(CandidateRef("summaries_knn@distance=whitened")).recipe(seed=1).distance
        is ScoreDistance.WHITENED
    )


@pytest.mark.parametrize(
    "name",
    [
        "pretext_error@distance=whitened",
        "frozen_knn@distance=cosine",
        "frozen_knn@distance=whitened,neighbours=10",
        "frozen_knn@neighbours=10,distance=whitened",
        "frozen_knn@neighbours=5,distance=whitened",
    ],
    ids=[
        "distance of an error",
        "no such distance",
        "neighbours after",
        "neighbours before",
        "neighbours as they are",
    ],
)
def test_a_distance_the_score_does_not_have_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        ASKING.arm_of(CandidateRef(name))


def test_a_detector_reading_its_errors_per_channel_says_so() -> None:
    turned = ASKING.arm_of(CandidateRef("pretext_error@error=channel"))
    stated = {
        p.name: p.value
        for p in ASKING.describe(CandidateRef("pretext_error@error=channel")).method.parameters
    }

    assert turned.recipe(seed=1).normalisation is ErrorNormalisation.PER_CHANNEL
    assert stated["error"] == "channel"
    # A campaign declared before the reading was a knob keeps the design it was stored with.
    assert "error" not in {
        p.name for p in ASKING.describe(CandidateRef("pretext_error")).method.parameters
    }


@pytest.mark.parametrize(
    "name",
    ["summaries_knn@error=channel", "pretext_error@error=quantile"],
    ids=["errors of a distance", "no such reading"],
)
def test_a_reading_of_errors_the_score_does_not_have_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        ASKING.arm_of(CandidateRef(name))


def test_a_detector_asked_for_the_tail_says_so_and_asks_once() -> None:
    turned = ASKING.arm_of(CandidateRef("pretext_error@hidden=tail"))
    stated = {
        p.name: p.value
        for p in ASKING.describe(CandidateRef("pretext_error@hidden=tail")).method.parameters
    }

    assert turned.recipe(seed=1).asked().hidden is HiddenCellsChoice.TAIL
    assert turned.recipe(seed=1).asked().draws == 1
    assert stated["hidden"] == "tail"
    assert stated["draws"] == "1"
    assert "hidden" not in {
        p.name for p in ASKING.describe(CandidateRef("pretext_error")).method.parameters
    }


@pytest.mark.parametrize(
    "name",
    [
        "pretext_error@draws=4,hidden=tail",
        "pretext_error@draws=8,hidden=tail",
        "pretext_error@hidden=tail,draws=4",
        "pretext_error@hidden=middle",
        "summaries_knn@hidden=tail",
    ],
    ids=[
        "draws before",
        "draws as they are",
        "draws after",
        "no such cells",
        "cells of a distance",
    ],
)
def test_a_hiding_the_score_does_not_have_is_refused(name: str) -> None:
    with pytest.raises(UnknownCandidateError):
        ASKING.arm_of(CandidateRef(name))


@pytest.mark.parametrize("distance", ["centre", "farthest", "subspace"])
def test_a_bar_of_the_field_names_its_distance_and_no_neighbours(distance: str) -> None:
    cells = DetectorArm(
        ref=CandidateRef("cells_knn"),
        features=DetectorFeatures.RAW_CELLS,
        architecture=None,
        backbone=None,
    )
    bars = DetectorCatalogue((cells,))
    stated = {
        p.name: p.value
        for p in bars.describe(CandidateRef(f"cells_knn@distance={distance}")).method.parameters
    }

    assert stated["distance"] == distance
    assert stated["features"] == "raw_cells"
    assert "neighbours" not in stated
    assert (
        bars.arm_of(CandidateRef(f"cells_knn@distance={distance}")).recipe(seed=1).neighbours
        is None
    )
    with pytest.raises(UnknownCandidateError):
        bars.arm_of(CandidateRef(f"cells_knn@distance={distance},neighbours=3"))


def test_a_detector_taught_events_names_its_count_and_head_and_no_distance_or_neighbours() -> None:
    taught = CATALOGUE.arm_of(CandidateRef("frozen_knn@events=5"))
    trees = CATALOGUE.arm_of(CandidateRef("summaries_knn@events=3,head=trees,reference=unit"))
    described = CATALOGUE.describe(CandidateRef("summaries_knn@events=3,head=trees"))
    stated = {p.name: p.value for p in described.method.parameters}

    assert taught.events == 5
    assert taught.recipe(seed=1).neighbours is None
    assert taught.recipe(seed=1).is_taught
    assert trees.recipe(seed=2).head is TaughtHead.TREES
    assert (stated["events"], stated["head"]) == ("3", "trees")
    assert {"neighbours", "distance"}.isdisjoint(stated)
    assert "head" not in {
        p.name for p in CATALOGUE.describe(CandidateRef("frozen_knn@events=5")).method.parameters
    }


@pytest.mark.parametrize(
    "name",
    [
        "frozen_knn@events=5,distance=whitened",
        "frozen_knn@distance=whitened,events=5",
        "frozen_knn@events=5,neighbours=3",
        "frozen_knn@neighbours=3,events=5",
        "frozen_knn@head=trees,events=5",
        "frozen_knn@head=trees",
        "frozen_knn@events=5,head=direction",
        "frozen_knn@events=0",
        "frozen_knn@events=5,head=nearest",
        "pretext_error@events=5",
    ],
)
def test_a_teaching_the_detector_cannot_take_is_refused(name: str) -> None:
    catalogue = DetectorCatalogue((FROZEN, UNTRAINED, SUMMARIES, PRETEXT))

    with pytest.raises(UnknownCandidateError):
        catalogue.arm_of(CandidateRef(name))
