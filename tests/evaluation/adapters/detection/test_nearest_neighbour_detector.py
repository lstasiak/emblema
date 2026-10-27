import numpy as np
import pytest

from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)

# Three fitting rows on a line, so every distance is known by hand.
FITTED = np.array([[0.0], [1.0], [2.0]])


def test_a_small_fitting_side_keeps_the_neighbours_it_could_score_itself_against() -> None:
    detector = NearestNeighbourDetector.fitted_on(FITTED, neighbours=5, contamination=0.0)

    assert detector.neighbours == 2
    # The row at 1 is one unit from both others; the ends average one and two in the
    # standardised units, so the alarm at the top quantile is the ends' own score.
    scale = float(np.std(FITTED))
    assert detector.alarm == pytest.approx(1.5 / scale)
    # A new row at the centre is read against the same two neighbours the alarm was set over:
    # the middle row at no distance and one end a unit away.
    assert detector.scores(np.array([[1.0]])) == pytest.approx([0.5 / scale - 1.5 / scale])


def test_the_kept_form_scores_as_the_fitted_one() -> None:
    detector = NearestNeighbourDetector.fitted_on(FITTED, neighbours=5, contamination=0.0)
    rows = np.array([[1.0], [10.0]])

    restored = NearestNeighbourDetector.from_bytes(detector.to_bytes())

    assert restored.neighbours == detector.neighbours
    assert restored.scores(rows) == pytest.approx(detector.scores(rows))


def test_a_lone_fitting_row_scores_zero_on_itself() -> None:
    detector = NearestNeighbourDetector.fitted_on(
        np.array([[3.0]]), neighbours=5, contamination=0.0
    )

    assert detector.neighbours == 1
    assert detector.alarm == 0.0
