import numpy as np
import pytest

from emblema.evaluation.adapters.minirocket.fitted_convolutions import FittedConvolutions
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
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
        recipe(method=convolutions()), convolutions(), 2, 16, WINDOWS, TARGETS, 10.0
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
