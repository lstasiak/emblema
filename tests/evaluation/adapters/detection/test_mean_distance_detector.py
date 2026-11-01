import numpy as np
import pytest

from emblema.evaluation.adapters.detection.mean_distance_detector import MeanDistanceDetector
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.exceptions import (
    InvalidDetectorRecipeError,
    UnreadableFittedCandidateError,
)

FITTED = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
# The fitting mean is (1, 1) and every feature's spread 1, so a row's standardised departures
# are its coordinates less one.
ROWS = np.array([[1.0, 1.0], [4.0, 1.0], [3.0, 3.0]])


def test_the_centre_distance_is_the_squared_standardised_distance_from_the_mean() -> None:
    detector = MeanDistanceDetector.fitted_on(
        FITTED, distance=ScoreDistance.CENTRE, contamination=0.25
    )

    # Every fitting row lies at squared distance 2, so the alarm sits there.
    assert detector.alarm.level == pytest.approx(2.0)
    assert detector.scores(ROWS) == pytest.approx([0.0 - 2.0, 9.0 - 2.0, 8.0 - 2.0])


def test_the_farthest_distance_is_the_largest_departure_of_one_feature() -> None:
    detector = MeanDistanceDetector.fitted_on(
        FITTED, distance=ScoreDistance.FARTHEST, contamination=0.25
    )

    assert detector.alarm.level == pytest.approx(1.0)
    assert detector.scores(ROWS) == pytest.approx([0.0 - 1.0, 3.0 - 1.0, 2.0 - 1.0])


def test_the_two_distances_rank_a_departure_in_one_feature_and_in_many_otherwise() -> None:
    centre = MeanDistanceDetector.fitted_on(
        FITTED, distance=ScoreDistance.CENTRE, contamination=0.25
    )
    farthest = MeanDistanceDetector.fitted_on(
        FITTED, distance=ScoreDistance.FARTHEST, contamination=0.25
    )
    # Departures of (3, 0) and (2.2, 2.2): the first farther in one feature, the second farther
    # in all.
    one_far, many_near = np.array([[4.0, 1.0]]), np.array([[3.2, 3.2]])

    assert centre.scores(many_near) > centre.scores(one_far)
    assert farthest.scores(one_far) > farthest.scores(many_near)


def test_a_distance_that_is_not_from_the_mean_is_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="from the mean"):
        MeanDistanceDetector.fitted_on(FITTED, distance=ScoreDistance.WHITENED, contamination=0.25)


def test_the_kept_form_scores_as_the_fitted_one_and_remembers_its_distance() -> None:
    detector = MeanDistanceDetector.fitted_on(
        FITTED, distance=ScoreDistance.FARTHEST, contamination=0.25
    )

    restored = MeanDistanceDetector.from_bytes(detector.to_bytes())

    assert restored.distance is ScoreDistance.FARTHEST
    assert restored.scores(ROWS) == pytest.approx(detector.scores(ROWS))
    assert detector.kept().format == MeanDistanceDetector.FORMAT
    with pytest.raises(UnreadableFittedCandidateError):
        MeanDistanceDetector.from_bytes(b"not an archive")


def test_a_distance_read_against_its_unit_is_its_place_among_the_units_own() -> None:
    units = ["a"] * 4 + ["b"] * 4
    detector = MeanDistanceDetector.fitted_on(
        np.vstack((FITTED, FITTED + 20.0)),
        distance=ScoreDistance.CENTRE,
        contamination=0.25,
        units=units,
    )

    read = detector.scores(FITTED + 40.0, units=["c"] * 4)

    assert detector.alarm.per_unit
    assert np.median(read) == pytest.approx(-detector.alarm.level, abs=1e-9)
