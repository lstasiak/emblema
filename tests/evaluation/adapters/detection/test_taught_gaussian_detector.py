import numpy as np
import pytest

from emblema.evaluation.adapters.detection.taught_gaussian_detector import TaughtGaussianDetector
from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import (
    InvalidDetectorRecipeError,
    InvalidTaughtEventsError,
    UnreadableFittedCandidateError,
)

# Fitting rows spread widely along the diagonal of two features and six times less across it,
# and the taught windows of two events displaced across it, where the fitting rows rarely go.
# The shrinkage lifts the quiet direction's variance a little, so the spread across is not made
# tiny: a whitened reading of a direction of nought spread is the shrinkage's, not the data's.
DRAWS = np.random.default_rng(11)
NORMAL = DRAWS.normal(size=(400, 1)) * np.array([[3.0, 3.0]]) + DRAWS.normal(size=(400, 2)) * 0.5
ACROSS = np.array([2.0, -2.0])
TAUGHT = np.vstack((NORMAL[:3] + ACROSS, NORMAL[3:4] + ACROSS))
ROWS = np.vstack((NORMAL, TAUGHT))
EVENTS = TaughtRows(positions=(400, 401, 402, 403), events=(1, 1, 1, 2))


def fitted(head: TaughtHead = TaughtHead.DIRECTION) -> TaughtGaussianDetector:
    return TaughtGaussianDetector.fitted_on(ROWS, EVENTS, head=head, contamination=0.05)


def test_the_direction_is_whitened_so_a_quiet_displacement_the_events_share_scores_high() -> None:
    detector = fitted()
    along = NORMAL[:1] + np.array([[3.0, 3.0]])
    across = NORMAL[:1] + ACROSS

    # Far along the loud direction the fitting rows cover is nothing to the direction; the small
    # displacement the events share is many whitened spreads out.
    assert detector.scores(across) > detector.scores(along) + 3.0
    assert detector.scores(NORMAL[:1] - ACROSS) < detector.scores(NORMAL[:1])


def test_the_fitting_rows_spread_by_about_one_along_the_direction() -> None:
    detector = fitted()
    along = detector.standardisation.of(NORMAL) @ detector.axis

    assert abs(float(along.mean())) < 0.2
    assert float(along.std()) == pytest.approx(1.0, rel=0.2)


def test_the_mean_and_covariance_come_from_the_rows_outside_the_events() -> None:
    detector = fitted()
    outside = TaughtGaussianDetector.fitted_on(
        np.vstack((NORMAL, NORMAL[:4] + 50.0)),
        EVENTS,
        head=TaughtHead.DIRECTION,
        contamination=0.05,
    )

    assert detector.standardisation.centre == pytest.approx(NORMAL.mean(axis=0))
    assert outside.standardisation.centre == pytest.approx(NORMAL.mean(axis=0))


def test_every_event_weighs_one_however_many_windows_it_spans() -> None:
    # Over round fitting rows, one event of three windows stands out in the first feature and
    # one of a single window in the second, by the same amount: weighed per event the direction
    # lies between the two features, weighed per window it leans to the first.
    draws = np.random.default_rng(12)
    round_rows = draws.normal(size=(400, 2))
    taught = np.array([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]]) + round_rows.mean(axis=0)
    rows = np.vstack((round_rows, taught))
    per_event = TaughtGaussianDetector.fitted_on(
        rows, EVENTS, head=TaughtHead.DIRECTION, contamination=0.05
    )
    per_window = TaughtGaussianDetector.fitted_on(
        rows,
        TaughtRows(positions=(400, 401, 402, 403), events=(1, 2, 3, 4)),
        head=TaughtHead.DIRECTION,
        contamination=0.05,
    )

    assert per_event.axis[0] == pytest.approx(per_event.axis[1], rel=0.15)
    assert per_window.axis[0] > 2.0 * per_window.axis[1]


def test_the_combination_raises_a_row_far_off_the_direction_which_the_direction_does_not() -> None:
    direction = fitted()
    combined = fitted(TaughtHead.COMBINED)
    far_along = NORMAL[:1] + np.array([[30.0, 30.0]])  # ten fitting spreads along the diagonal

    assert direction.scores(far_along) < 1.0
    assert combined.scores(far_along) > combined.scores(NORMAL[:1]) + 3.0
    assert combined.scales is not None
    assert direction.scales is None


def test_events_at_the_fitting_mean_give_no_direction() -> None:
    rows = np.vstack((NORMAL, np.tile(NORMAL.mean(axis=0), (2, 1))))

    with pytest.raises(InvalidTaughtEventsError, match="no direction"):
        TaughtGaussianDetector.fitted_on(
            rows,
            TaughtRows(positions=(400, 401), events=(1, 1)),
            head=TaughtHead.DIRECTION,
            contamination=0.05,
        )


def test_trees_are_no_head_of_a_gaussian_and_too_few_rows_outside_the_events_are_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="scores by one of"):
        fitted(TaughtHead.TREES)
    with pytest.raises(ValueError, match="two fitting rows outside"):
        TaughtGaussianDetector.fitted_on(
            ROWS[399:],
            TaughtRows(positions=(1, 2, 3, 4), events=(1, 1, 1, 2)),
            head=TaughtHead.DIRECTION,
            contamination=0.05,
        )


@pytest.mark.parametrize("head", [TaughtHead.DIRECTION, TaughtHead.COMBINED])
def test_the_kept_form_scores_as_the_fitted_one(head: TaughtHead) -> None:
    detector = fitted(head)
    rows = np.array([[1.0, 1.0], [4.0, -4.0], [0.3, -0.3]])

    restored = TaughtGaussianDetector.from_bytes(detector.to_bytes())

    assert restored.scores(rows) == pytest.approx(detector.scores(rows))
    assert restored.head is head
    assert restored.scales == detector.scales
    assert detector.kept().format == TaughtGaussianDetector.FORMAT
    with pytest.raises(UnreadableFittedCandidateError):
        TaughtGaussianDetector.from_bytes(b"not an archive")


def test_a_detector_read_against_its_unit_is_read_against_the_units_own_scores() -> None:
    units = ["a"] * 200 + ["b"] * 200 + ["a"] * 4
    detector = TaughtGaussianDetector.fitted_on(
        ROWS, EVENTS, head=TaughtHead.DIRECTION, contamination=0.05, units=units
    )

    read = detector.scores(NORMAL[:10], units=["c"] * 10)

    assert detector.alarm.per_unit
    assert np.all(np.isfinite(read))
