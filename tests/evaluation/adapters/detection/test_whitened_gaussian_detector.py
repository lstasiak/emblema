import numpy as np
import pytest

from emblema.evaluation.adapters.detection.whitened_gaussian_detector import (
    WhitenedGaussianDetector,
)
from emblema.evaluation.domain.exceptions import (
    InvalidScoredOutcomeError,
    UnreadableFittedCandidateError,
)

# Two features that move together: the fitting rows lie along the diagonal, so the direction
# across it carries almost none of their variation.
DRAWS = np.random.default_rng(3)
ALONG = DRAWS.normal(size=(400, 1)) * np.array([[3.0, 3.0]])
FITTED = ALONG + DRAWS.normal(size=(400, 2)) * 0.1


def test_the_whitened_distance_weighs_a_quiet_direction_as_much_as_a_loud_one() -> None:
    detector = WhitenedGaussianDetector.fitted_on(FITTED, contamination=0.02)
    # One fitting spread along the diagonal, and one fitting spread across it: forty times
    # shorter in the standardised coordinates, the same length once whitened.
    along = np.array([[3.0, 3.0]])
    across = np.array([[0.1, -0.1]]) / np.sqrt(2.0)
    standardised = detector.standardisation.of

    assert np.linalg.norm(standardised(along)) > 10 * np.linalg.norm(standardised(across))
    assert detector.scores(across) == pytest.approx(detector.scores(along), rel=0.25)


def test_the_score_is_the_squared_whitened_length_above_the_alarm() -> None:
    rows = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    detector = WhitenedGaussianDetector.fitted_on(rows, contamination=0.25)
    standardised = detector.standardisation.of(rows)
    expected = np.einsum("ij,ij->i", standardised @ detector.root, standardised @ detector.root)

    assert detector.scores(rows) == pytest.approx(expected - detector.alarm.level)
    assert detector.alarm.level == pytest.approx(np.quantile(expected, 0.75))


def test_a_feature_without_spread_is_left_as_it_is() -> None:
    rows = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

    detector = WhitenedGaussianDetector.fitted_on(rows, contamination=0.1)

    assert detector.standardisation.scale[1] == 1.0
    assert np.all(np.isfinite(detector.scores(np.array([[2.0, 7.0]]))))


def test_the_kept_form_scores_as_the_fitted_one() -> None:
    detector = WhitenedGaussianDetector.fitted_on(FITTED, contamination=0.05)
    rows = np.array([[1.0, 1.0], [4.0, -4.0]])

    restored = WhitenedGaussianDetector.from_bytes(detector.to_bytes())

    assert restored.scores(rows) == pytest.approx(detector.scores(rows))
    assert restored.alarm == detector.alarm


def test_bytes_that_are_no_archive_are_refused() -> None:
    with pytest.raises(UnreadableFittedCandidateError):
        WhitenedGaussianDetector.from_bytes(b"not an archive")


def test_a_distance_read_against_its_unit_is_its_place_among_the_units_own() -> None:
    units = ["a"] * 200 + ["b"] * 200
    shifted = FITTED + np.r_[np.zeros((200, 2)), np.full((200, 2), 20.0)]
    detector = WhitenedGaussianDetector.fitted_on(shifted, contamination=0.1, units=units)
    drifted = FITTED[:50] + 40.0

    read = detector.scores(drifted, units=["c"] * 50)

    assert detector.alarm.per_unit
    # A month that drifted whole reads as ordinary against itself.
    assert np.median(read) == pytest.approx(-detector.alarm.level, abs=1e-9)
    with pytest.raises(ValueError, match="given no units"):
        detector.scores(drifted)


def test_units_none_of_which_holds_two_windows_are_refused() -> None:
    with pytest.raises(InvalidScoredOutcomeError, match="no unit holds two"):
        WhitenedGaussianDetector.fitted_on(FITTED[:3], contamination=0.1, units=["a", "b", "c"])


def test_fewer_than_two_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="two rows"):
        WhitenedGaussianDetector.fitted_on(FITTED[:1], contamination=0.1)


def test_fitting_rows_alike_in_every_feature_still_score_finitely() -> None:
    alike = np.tile([[1.0, 2.0, 3.0, 4.0]], (20, 1))

    detector = WhitenedGaussianDetector.fitted_on(alike, contamination=0.1)

    assert np.all(np.isfinite(detector.scores(np.array([[1.0, 2.0, 3.0, 9.0]]))))
