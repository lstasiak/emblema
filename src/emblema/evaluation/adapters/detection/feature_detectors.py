from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.feature_detector import FeatureDetector
from emblema.evaluation.adapters.detection.mean_distance_detector import MeanDistanceDetector
from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)
from emblema.evaluation.adapters.detection.subspace_detector import SubspaceDetector
from emblema.evaluation.adapters.detection.taught_gaussian_detector import TaughtGaussianDetector
from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.adapters.detection.taught_trees_detector import TaughtTreesDetector
from emblema.evaluation.adapters.detection.whitened_gaussian_detector import (
    WhitenedGaussianDetector,
)
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import InvalidTaughtEventsError


class FeatureDetectors:
    """The detectors over features, one per distance a recipe can score by or head it is taught.

    The one place that turns a recipe's distance, or the head of a recipe taught events, into a
    detector, so the runtime over an encoder's states and the one over the summaries fit the
    same detector for the same name.
    """

    @classmethod
    def fitted(
        cls,
        recipe: DetectorRecipe,
        rows: NDArray[np.float64],
        *,
        units: Sequence[str] | None,
        taught: TaughtRows | None = None,
    ) -> FeatureDetector:
        """The detector ``recipe`` names, fitted on ``rows``, one per fitting window.

        Raises:
            InvalidDetectorRecipeError: If the recipe scores by an error, not a distance.
            InvalidTaughtEventsError: If the recipe is taught events and ``taught`` names none or
                another count of them, or it is taught none and ``taught`` names some.
        """
        if recipe.is_taught:
            if taught is None:
                raise InvalidTaughtEventsError(
                    f"a detector taught {recipe.events} events was handed no taught window"
                )
            if taught.event_count != recipe.events:
                raise InvalidTaughtEventsError(
                    f"a detector taught {recipe.events} events was handed {taught.event_count}"
                )
            if recipe.head is TaughtHead.TREES:
                return TaughtTreesDetector.fitted_on(
                    rows, taught, seed=recipe.seed, contamination=recipe.contamination, units=units
                )
            return TaughtGaussianDetector.fitted_on(
                rows, taught, head=recipe.head, contamination=recipe.contamination, units=units
            )
        if taught is not None:
            raise InvalidTaughtEventsError("a detector taught no event was handed taught windows")
        if recipe.distance is ScoreDistance.WHITENED:
            return WhitenedGaussianDetector.fitted_on(
                rows, contamination=recipe.contamination, units=units
            )
        if recipe.distance is ScoreDistance.SUBSPACE:
            return SubspaceDetector.fitted_on(rows, contamination=recipe.contamination, units=units)
        if recipe.distance in MeanDistanceDetector.DISTANCES:
            return MeanDistanceDetector.fitted_on(
                rows, distance=recipe.distance, contamination=recipe.contamination, units=units
            )
        return NearestNeighbourDetector.fitted_on(
            rows, neighbours=recipe.nearest(), contamination=recipe.contamination, units=units
        )
