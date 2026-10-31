import numpy as np
import pytest

from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)
from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError

# Three fitting rows on a line, so every distance is known by hand.
FITTED = np.array([[0.0], [1.0], [2.0]])


def test_a_small_fitting_side_keeps_the_neighbours_it_could_score_itself_against() -> None:
    detector = NearestNeighbourDetector.fitted_on(FITTED, neighbours=5, contamination=0.0)

    assert detector.neighbours == 2
    # The row at 1 is one unit from both others; the ends average one and two in the
    # standardised units, so the alarm at the top quantile is the ends' own score.
    scale = float(np.std(FITTED))
    assert detector.alarm.level == pytest.approx(1.5 / scale)
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
    assert detector.alarm.level == 0.0


# Two units of fitting rows, the second shifted far from the first: read against the fitting
# side as a whole, every row of the second unit lies far from half of the fit; read against its
# own unit, it lies where the first unit's rows lie in theirs.
SHIFTED = np.array([[0.0], [0.1], [0.2], [0.3], [10.0], [10.1], [10.2], [10.3]])
SHIFTED_UNITS = ["a"] * 4 + ["b"] * 4


def test_a_distance_read_against_its_unit_is_its_place_among_the_units_own() -> None:
    detector = NearestNeighbourDetector.fitted_on(
        SHIFTED, neighbours=1, contamination=0.25, units=SHIFTED_UNITS
    )
    # A month that drifted as a whole, with one window far out within it.
    drifted = np.array([[50.0], [50.1], [50.2], [50.3], [55.0]])

    read = detector.scores(drifted, units=["c"] * 5)

    assert detector.alarm.per_unit
    # The drifted bulk sits at its own unit's median; the far window alone stands out.
    assert int(np.argmax(read)) == 4
    assert np.median(read[:4] + detector.alarm.level) == pytest.approx(0.0, abs=1.0)
    # Against the fitting side, every window of the drifted month raises an alarm.
    against_the_fit = NearestNeighbourDetector.fitted_on(SHIFTED, neighbours=1, contamination=0.25)
    assert (against_the_fit.scores(drifted) > 0.0).all()


def test_a_unit_without_a_spread_of_its_own_is_read_over_the_fitting_side_s() -> None:
    detector = NearestNeighbourDetector.fitted_on(
        SHIFTED, neighbours=1, contamination=0.25, units=SHIFTED_UNITS
    )
    # A lone window, and a unit of windows alike to the rounding of the distances.
    rows = np.array([[99.0], [5.0], [5.0 + 1e-12], [5.0 - 1e-12], [20.0]])

    read = detector.scores(rows, units=["d", "e", "e", "e", "e"])

    assert detector.alarm.unit_spread > 0.0
    assert read[0] == pytest.approx(-detector.alarm.level)
    # The windows alike read at their median, and the one apart over the borrowed spread, not
    # over a spread of rounding that would put it a billion spreads out.
    assert read[1:4] + detector.alarm.level == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert 0.0 < read[4] + detector.alarm.level < 100.0


def test_units_none_of_which_holds_two_windows_are_refused() -> None:
    detector = NearestNeighbourDetector.fitted_on(
        SHIFTED, neighbours=1, contamination=0.25, units=SHIFTED_UNITS
    )

    with pytest.raises(InvalidScoredOutcomeError, match="no unit holds two windows"):
        detector.scores(np.array([[1.0], [2.0]]), units=["x", "y"])
    with pytest.raises(InvalidScoredOutcomeError, match="no unit holds two windows"):
        NearestNeighbourDetector.fitted_on(
            FITTED, neighbours=1, contamination=0.25, units=["a", "b", "c"]
        )


def test_a_detector_reading_units_refuses_rows_without_them_and_the_other_way_round() -> None:
    per_unit = NearestNeighbourDetector.fitted_on(
        SHIFTED, neighbours=1, contamination=0.25, units=SHIFTED_UNITS
    )
    whole = NearestNeighbourDetector.fitted_on(SHIFTED, neighbours=1, contamination=0.25)

    with pytest.raises(ValueError, match="given no units"):
        per_unit.scores(SHIFTED)
    with pytest.raises(ValueError, match="reads no unit"):
        whole.scores(SHIFTED, units=SHIFTED_UNITS)
    with pytest.raises(ValueError, match="units were given"):
        per_unit.scores(SHIFTED, units=["a"])
    with pytest.raises(ValueError, match="units were given"):
        NearestNeighbourDetector.fitted_on(SHIFTED, neighbours=1, contamination=0.25, units=["a"])


def test_the_kept_form_remembers_that_it_reads_units() -> None:
    detector = NearestNeighbourDetector.fitted_on(
        SHIFTED, neighbours=1, contamination=0.25, units=SHIFTED_UNITS
    )

    restored = NearestNeighbourDetector.from_bytes(detector.to_bytes())

    assert restored.alarm.per_unit
    assert restored.alarm.unit_spread == detector.alarm.unit_spread
    assert restored.scores(SHIFTED, units=SHIFTED_UNITS) == pytest.approx(
        detector.scores(SHIFTED, units=SHIFTED_UNITS)
    )


def test_a_detector_kept_before_units_were_read_reads_none() -> None:
    import io

    buffer = io.BytesIO()
    np.savez(
        buffer,
        centre=np.zeros(1),
        scale=np.ones(1),
        fitted=FITTED,
        neighbours=np.asarray(1),
        alarm=np.asarray(0.5),
    )

    restored = NearestNeighbourDetector.from_bytes(buffer.getvalue())
    assert (restored.alarm.per_unit, restored.alarm.unit_spread) == (False, 1.0)
