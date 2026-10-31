from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.feature_detector import FeatureDetector
from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)
from emblema.evaluation.adapters.detection.whitened_gaussian_detector import (
    WhitenedGaussianDetector,
)
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.score_distance import ScoreDistance


class FeatureDetectors:
    """The detectors over features, one per distance a recipe can score by.

    The one place that turns a recipe's distance into a detector, so the runtime over an
    encoder's states and the one over the summaries fit the same detector for the same name.
    """

    @classmethod
    def fitted(
        cls, recipe: DetectorRecipe, rows: NDArray[np.float64], *, units: Sequence[str] | None
    ) -> FeatureDetector:
        """The detector ``recipe`` names, fitted on ``rows``, one per fitting window.

        Raises:
            InvalidDetectorRecipeError: If the recipe scores by an error, not a distance.
        """
        if recipe.distance is ScoreDistance.WHITENED:
            return WhitenedGaussianDetector.fitted_on(
                rows, contamination=recipe.contamination, units=units
            )
        return NearestNeighbourDetector.fitted_on(
            rows, neighbours=recipe.nearest(), contamination=recipe.contamination, units=units
        )
