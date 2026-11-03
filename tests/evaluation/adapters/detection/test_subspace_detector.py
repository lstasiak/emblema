import numpy as np
import pytest

from emblema.evaluation.adapters.detection.subspace_detector import SubspaceDetector
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError

# Fitting rows along the diagonal of two features, with a little spread across it: one leading
# component carries almost all the variance, so the subspace is the diagonal.
DRAWS = np.random.default_rng(5)
FITTED = DRAWS.normal(size=(500, 1)) * np.array([[3.0, 3.0]]) + DRAWS.normal(size=(500, 2)) * 0.1


def test_the_subspace_is_spanned_by_the_components_that_carry_most_of_the_variance() -> None:
    detector = SubspaceDetector.fitted_on(FITTED, contamination=0.02)

    assert detector.components.shape == (2, 1)
    assert abs(detector.components[0, 0]) == pytest.approx(abs(detector.components[1, 0]), rel=0.05)


def test_a_row_far_along_the_subspace_and_one_far_off_it_both_score_high() -> None:
    detector = SubspaceDetector.fitted_on(FITTED, contamination=0.02)
    along = np.array([[9.0, 9.0]])
    off = np.array([[0.5, -0.5]])
    ordinary = FITTED[:1]

    assert detector.scores(along) > detector.scores(ordinary) + 1.0
    assert detector.scores(off) > detector.scores(ordinary) + 1.0


def test_the_within_and_the_residual_statistics_are_read_over_their_fitting_medians() -> None:
    detector = SubspaceDetector.fitted_on(FITTED, contamination=0.5)
    standardised = detector.standardisation.of(FITTED)
    within, outside = SubspaceDetector._statistics(
        standardised, detector.components, detector.variances
    )

    assert detector.scales[0] == pytest.approx(np.median(within))
    # The residual's spread here is a sliver of the variance left outside the diagonal, so its
    # scale is the floor rather than the median.
    assert detector.scales[1] >= np.median(outside)
    # A row at the median of both statistics scores one for each.
    at_medians = SubspaceDetector._combined(
        np.array([detector.scales[0]]), np.array([detector.scales[1]]), detector.scales
    )
    assert at_medians == pytest.approx([2.0])


def test_the_kept_form_scores_as_the_fitted_one() -> None:
    detector = SubspaceDetector.fitted_on(FITTED, contamination=0.05)
    rows = np.array([[1.0, 1.0], [4.0, -4.0]])

    restored = SubspaceDetector.from_bytes(detector.to_bytes())

    assert restored.scores(rows) == pytest.approx(detector.scores(rows))
    assert restored.scales == detector.scales
    assert detector.kept().format == SubspaceDetector.FORMAT
    with pytest.raises(UnreadableFittedCandidateError):
        SubspaceDetector.from_bytes(b"not an archive")


def test_fewer_than_two_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="two rows"):
        SubspaceDetector.fitted_on(FITTED[:1], contamination=0.1)


def test_fewer_fitting_rows_than_features_keep_no_more_components_than_they_span() -> None:
    draws = np.random.default_rng(9)
    few = draws.normal(size=(20, 100))
    detector = SubspaceDetector.fitted_on(few, contamination=0.1)
    ordinary = draws.normal(size=(50, 100))
    far = ordinary.copy()
    far[:, 0] += 8.0

    assert detector.components.shape[1] <= 19
    assert np.all(np.isfinite(detector.scores(ordinary)))
    # The residual outside the subspace is read over a tenth of the variance left there, so a
    # fresh row scores within an order of magnitude of ten rather than off the scale.
    assert np.abs(detector.scores(ordinary)).max() < 100.0
    assert detector.scores(far).mean() > detector.scores(ordinary).mean()
