import io

import numpy as np
import pytest

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.evaluation.adapters.minirocket.fitted_convolutions import FittedConvolutions
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.evaluation.domain.heads.outcome_folds import OutcomeFolds
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.shared.kernel.tokens import TokenWindow
from tests.evaluation.adapters.features.support import timed, window
from tests.evaluation.support import convolutions, recipe

RNG = np.random.default_rng(5)


def wave(cycles: float) -> TokenWindow:
    times = np.sort(RNG.uniform(0.0, 1.0, 50))
    return window(
        timed(1, [(float(np.sin(2 * np.pi * cycles * t)), float(t)) for t in times]),
        timed(2, [(float(t), float(t)) for t in times[::3]]),
    )


WINDOWS = [wave(cycles) for cycles in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) * 3]
TARGETS = np.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0) * 3) / 10.0


def fit() -> FittedConvolutions:
    return FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        2,
        16,
        WINDOWS,
        TARGETS,
        10.0,
        TargetKind.CONTINUOUS,
    )


def test_answers_come_back_in_the_task_unit() -> None:
    answered = fit().predict(WINDOWS, threads=1)

    assert np.corrcoef(answered, TARGETS * 10.0)[0, 1] > 0.9


def test_what_was_kept_answers_exactly_as_what_was_fitted() -> None:
    fitted = fit()

    read = FittedConvolutions.read(fitted.to_bytes())

    assert np.array_equal(read.predict(WINDOWS, threads=1), fitted.predict(WINDOWS, threads=1))
    assert read.parameters == fitted.parameters
    assert read.penalty in convolutions().ridge.penalties


def test_bytes_that_are_not_fitted_convolutions_are_refused() -> None:
    with pytest.raises(UnreadableFittedCandidateError):
        FittedConvolutions.read(b"not an archive")


def test_the_grid_reads_the_channels_the_fitted_windows_hold_and_no_other() -> None:
    # A vocabulary of five: the fitted windows hold channels 1 and 2, channel 1 at every step of
    # the grid and channel 2 at a third of them, so only channel 2's mask says anything.
    dense = [
        window(
            timed(1, [(float(np.sin(k)), (k + 0.5) / 16) for k in range(16)]),
            timed(2, [(float(k), (k + 0.5) / 16) for k in range(0, 16, 3)]),
        )
        for _ in range(6)
    ]

    fitted = FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        5,
        16,
        dense,
        np.linspace(0.0, 1.0, 6),
        1.0,
        TargetKind.CONTINUOUS,
    )

    assert fitted.rows.tolist() == [0, 1, 5 + 1]


def test_a_channel_no_fitted_window_held_is_not_read_when_answering() -> None:
    fitted = FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        5,
        16,
        WINDOWS,
        TARGETS,
        10.0,
        TargetKind.CONTINUOUS,
    )
    times = np.sort(np.random.default_rng(9).uniform(0.0, 1.0, 50))
    first = timed(1, [(float(np.sin(2 * np.pi * t)), float(t)) for t in times])
    second = timed(2, [(float(t), float(t)) for t in times[::3]])

    alone = fitted.predict([window(first, second)], threads=1)
    beside = fitted.predict([window(first, second, timed(4, [(9.0, 0.5)]))], threads=1)

    assert np.array_equal(alone, beside)


OUTCOMES = np.array((0.0, 0.0, 0.0, 1.0, 1.0, 1.0) * 3)


def fit_outcomes() -> FittedConvolutions:
    return FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        2,
        16,
        WINDOWS,
        OUTCOMES,
        1.0,
        TargetKind.BINARY,
    )


def features_of(fitted: FittedConvolutions) -> np.ndarray:
    laid = RegularGrid(fitted.grid_steps, fitted.channels).of(WINDOWS)[:, fitted.rows]
    return np.asarray(fitted.transform.of(laid) / fitted.feature_scale)


def test_a_binary_task_is_answered_with_the_probabilities_of_its_log_odds() -> None:
    fitted = fit_outcomes()

    answered = fitted.predict(WINDOWS, threads=1)

    log_odds = features_of(fitted) @ fitted.weights + fitted.intercept
    assert fitted.kind is TargetKind.BINARY
    assert np.all((answered > 0.0) & (answered < 1.0))
    assert np.allclose(answered, 1.0 / (1.0 + np.exp(-log_odds)))
    assert answered[OUTCOMES == 1.0].mean() > answered[OUTCOMES == 0.0].mean()


def test_what_was_kept_of_a_binary_task_answers_exactly_as_what_was_fitted() -> None:
    fitted = fit_outcomes()

    read = FittedConvolutions.read(fitted.to_bytes())

    assert read.kind is TargetKind.BINARY
    assert np.array_equal(read.predict(WINDOWS, threads=1), fitted.predict(WINDOWS, threads=1))


def test_a_candidate_kept_before_outcomes_were_answered_reads_as_a_quantity() -> None:
    fitted = fit()
    with np.load(io.BytesIO(fitted.to_bytes()), allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files if name != "kind"}
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)

    read = FittedConvolutions.read(buffer.getvalue())

    assert read.kind is TargetKind.CONTINUOUS
    assert np.array_equal(read.predict(WINDOWS, threads=1), fitted.predict(WINDOWS, threads=1))


def test_the_penalty_of_a_binary_task_gives_the_folds_the_smallest_log_loss() -> None:
    from sklearn.linear_model import LogisticRegression

    fitted = fit_outcomes()
    features = features_of(fitted)
    folds = OutcomeFolds.of(OUTCOMES.tolist())
    losses = []
    for penalty in convolutions().ridge.penalties:
        per_fold = []
        for fold in range(folds.count):
            kept, held = list(folds.kept(fold)), list(folds.held_out(fold))
            model = LogisticRegression(
                C=1.0 / penalty, solver="newton-cholesky", tol=1e-12, max_iter=1000
            ).fit(features[kept], OUTCOMES[kept])
            probability = model.predict_proba(features[held])[:, 1]
            truth = OUTCOMES[held]
            per_fold.append(
                -np.mean(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability))
            )
        losses.append(np.mean(per_fold))

    assert fitted.penalty == convolutions().ridge.penalties[int(np.argmin(losses))]


def test_outcomes_the_waves_barely_tell_apart_are_still_ranked_by_the_map() -> None:
    # Outcomes that follow no wave: a least-squares map calibrated on answers left out one at a
    # time reversed its ranking here; the likelihood's own map answers every window apart, and at
    # its optimum its answers lean towards the outcomes it was fitted on, never against them.
    outcomes = np.zeros(len(WINDOWS))
    outcomes[[0, 2, 11, 13]] = 1.0
    fitted = FittedConvolutions.fitted(
        recipe(method=convolutions()),
        convolutions(),
        2,
        16,
        WINDOWS,
        outcomes,
        1.0,
        TargetKind.BINARY,
    )

    answered = fitted.predict(WINDOWS, threads=1)

    assert np.unique(answered).size == len(np.unique(features_of(fitted), axis=0))
    assert answered[outcomes == 1.0].mean() > answered[outcomes == 0.0].mean()
