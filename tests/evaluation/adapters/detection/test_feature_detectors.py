from dataclasses import replace

import numpy as np
import pytest

from emblema.evaluation.adapters.detection.feature_detectors import FeatureDetectors
from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)
from emblema.evaluation.adapters.detection.whitened_gaussian_detector import (
    WhitenedGaussianDetector,
)
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError
from tests.evaluation.support import WEIGHTS

ROWS = np.random.default_rng(1).normal(size=(40, 3))
NEAREST = DetectorRecipe(
    features=DetectorFeatures.SUMMARIES, reading=None, neighbours=3, contamination=0.1, seed=1
)
WHITENED = replace(NEAREST, distance=ScoreDistance.WHITENED, neighbours=None)


def test_the_recipe_s_distance_chooses_the_detector() -> None:
    nearest = FeatureDetectors.fitted(NEAREST, ROWS, units=None)
    whitened = FeatureDetectors.fitted(WHITENED, ROWS, units=None)

    assert isinstance(nearest, NearestNeighbourDetector)
    assert nearest.neighbours == 3
    assert isinstance(whitened, WhitenedGaussianDetector)


def test_the_kept_form_carries_the_detector_s_own_format() -> None:
    for recipe, expected in (
        (NEAREST, NearestNeighbourDetector),
        (WHITENED, WhitenedGaussianDetector),
    ):
        kept = FeatureDetectors.fitted(recipe, ROWS, units=None).kept()

        assert kept.format == expected.FORMAT
        assert expected.from_bytes(kept.content).scores(ROWS[:2]) == pytest.approx(
            FeatureDetectors.fitted(recipe, ROWS, units=None).scores(ROWS[:2])
        )


def test_a_detector_scored_by_an_error_is_no_feature_detector() -> None:
    pretext = DetectorRecipe(
        features=DetectorFeatures.PRETEXT_ERROR,
        reading=None,
        neighbours=None,
        contamination=0.1,
        seed=1,
        pretext=PretextAsking(backbone=WEIGHTS, draws=2),
    )

    with pytest.raises(InvalidDetectorRecipeError, match="scores by an error"):
        FeatureDetectors.fitted(pretext, ROWS, units=None)
