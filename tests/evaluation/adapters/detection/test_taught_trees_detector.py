import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.adapters.detection.taught_trees_detector import (
    TaughtTreesDetector,
    TreeArrays,
)
from emblema.evaluation.domain.exceptions import (
    InvalidTaughtEventsError,
    UnreadableFittedCandidateError,
)

# Fitting rows over five features, and the windows of two events that stand out in the third
# feature alone, one of them a single window.
DRAWS = np.random.default_rng(21)
NORMAL = DRAWS.normal(size=(300, 5))
TAUGHT = DRAWS.normal(size=(4, 5))
TAUGHT[:, 2] += 4.0
ROWS = np.vstack((NORMAL, TAUGHT))
EVENTS = TaughtRows(positions=(300, 301, 302, 303), events=(1, 1, 1, 2))


def fitted(seed: int = 1) -> TaughtTreesDetector:
    return TaughtTreesDetector.fitted_on(ROWS, EVENTS, seed=seed, contamination=0.05)


def test_the_trees_tell_a_row_like_the_events_from_one_like_the_rest() -> None:
    detector = fitted()
    like_events = NORMAL[:5].copy()
    like_events[:, 2] += 4.0

    assert np.all(detector.scores(like_events) > detector.scores(NORMAL[:5]).max())


def test_a_single_window_event_is_learnt_as_much_as_a_longer_one() -> None:
    rows = ROWS.copy()
    # The lone event stands out in another feature than the three-window one does.
    rows[303, 2] = NORMAL[0, 2]
    rows[303, 4] += 4.0
    detector = TaughtTreesDetector.fitted_on(rows, EVENTS, seed=1, contamination=0.05)
    like_lone = NORMAL[:5].copy()
    like_lone[:, 4] += 4.0

    assert np.all(detector.scores(like_lone) > detector.scores(NORMAL[:5]).max())


def test_the_fit_repeats_under_a_seed_and_the_trees_are_read_as_the_library_reads_them() -> None:
    first, second = fitted(1), fitted(1)
    standardised = first.standardisation.of(ROWS)
    weights = np.ones(len(ROWS))
    weights[300:] = EVENTS.weights() * (300 / 2)
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=TaughtTreesDetector.ROUNDS,
        learning_rate=TaughtTreesDetector.LEARNING_RATE,
        max_depth=TaughtTreesDetector.DEPTH,
        max_features="sqrt",
        init="zero",
        random_state=1,
    ).fit(standardised, EVENTS.mask(len(ROWS)).astype(int), sample_weight=weights)

    assert first.scores(ROWS) == pytest.approx(second.scores(ROWS))
    assert TreeArrays.of(model).log_odds(standardised) == pytest.approx(
        model.decision_function(standardised)
    )
    assert first.trees.log_odds(standardised) == pytest.approx(
        model.decision_function(standardised)
    )


def test_a_value_at_a_threshold_falls_where_the_library_sends_it() -> None:
    detector = fitted()
    standardised = detector.standardisation.of(ROWS)
    trees = detector.trees
    splitting = np.flatnonzero(trees.left != -1)
    probes = standardised[:200].copy()
    # Every probe is put exactly on a split's threshold, in the feature that split reads.
    for row, node in enumerate(splitting[: len(probes)]):
        probes[row, trees.feature[node]] = trees.threshold[node]
    weights = np.ones(len(ROWS))
    weights[300:] = EVENTS.weights() * (300 / 2)
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=TaughtTreesDetector.ROUNDS,
        learning_rate=TaughtTreesDetector.LEARNING_RATE,
        max_depth=TaughtTreesDetector.DEPTH,
        max_features="sqrt",
        init="zero",
        random_state=1,
    ).fit(standardised, EVENTS.mask(len(ROWS)).astype(int), sample_weight=weights)

    assert trees.log_odds(probes) == pytest.approx(model.decision_function(probes))


def test_the_kept_form_scores_as_the_fitted_one() -> None:
    detector = fitted()

    restored = TaughtTreesDetector.from_bytes(detector.to_bytes())

    assert restored.scores(ROWS[:20]) == pytest.approx(detector.scores(ROWS[:20]))
    assert restored.alarm == detector.alarm
    assert detector.kept().format == TaughtTreesDetector.FORMAT
    with pytest.raises(UnreadableFittedCandidateError):
        TaughtTreesDetector.from_bytes(b"not an archive")


def test_rows_all_taught_or_taught_past_the_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="outside them"):
        TaughtTreesDetector.fitted_on(
            TAUGHT,
            TaughtRows(positions=(0, 1, 2, 3), events=(1, 1, 1, 2)),
            seed=1,
            contamination=0.1,
        )
    with pytest.raises(InvalidTaughtEventsError, match="past the"):
        TaughtTreesDetector.fitted_on(NORMAL, EVENTS, seed=1, contamination=0.1)


def test_a_detector_read_against_its_unit_answers_finite_scores() -> None:
    units = ["a"] * 150 + ["b"] * 150 + ["a"] * 4
    detector = TaughtTreesDetector.fitted_on(ROWS, EVENTS, seed=1, contamination=0.05, units=units)

    read = detector.scores(NORMAL[:10], units=["c"] * 10)

    assert detector.alarm.per_unit
    assert np.all(np.isfinite(read))
