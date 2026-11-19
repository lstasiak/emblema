"""The published forecaster's envelope: levels found, channels handed per window, chunks resumed."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.chronos2_envelope import envelope_of, level_indices, run

QUANTILES = (0.1, 0.2, 0.5, 0.8, 0.9)


def spread_around_last(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
    """A model standing in: each variate's quantiles are its last finite hour plus the level."""
    answers = []
    for series in batch:
        last = np.array([v[np.isfinite(v)][-1] for v in series], dtype=np.float32)
        levels = np.asarray(QUANTILES, dtype=np.float32)
        answers.append((last[:, None] + levels[None, :])[:, :, None])
    return answers


def test_the_three_levels_are_found_among_the_models() -> None:
    assert level_indices(QUANTILES) == [0, 2, 4]
    with pytest.raises(ValueError, match=r"no 0\.9"):
        level_indices((0.1, 0.5))


def test_each_window_hands_its_channels_with_a_reading_and_the_rest_stay_missing() -> None:
    contexts = np.full((2, 3, 4), np.nan, dtype=np.float32)
    contexts[0, 0] = [1.0, 2.0, np.nan, 3.0]
    contexts[0, 2] = [5.0, np.nan, np.nan, np.nan]
    contexts[1, 1] = [7.0, 7.0, 7.0, 8.0]

    envelope = envelope_of(contexts, spread_around_last, QUANTILES)

    assert envelope.shape == (2, 3, 3)
    assert envelope[0, 0].tolist() == pytest.approx([3.1, 3.5, 3.9])
    assert envelope[0, 2].tolist() == pytest.approx([5.1, 5.5, 5.9])
    assert envelope[1, 1].tolist() == pytest.approx([8.1, 8.5, 8.9])
    assert np.isnan(envelope[0, 1]).all()
    assert np.isnan(envelope[1, [0, 2]]).all()


def test_a_model_answering_other_variates_or_no_finite_value_is_refused() -> None:
    contexts = np.ones((1, 2, 3), dtype=np.float32)

    def one_variate(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        return [np.zeros((1, len(QUANTILES), 1), dtype=np.float32) for _ in batch]

    def missing(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        return [np.full((len(s), len(QUANTILES), 1), np.nan, dtype=np.float32) for s in batch]

    with pytest.raises(ValueError, match="answered 1 variates for 2"):
        envelope_of(contexts, one_variate, QUANTILES)
    with pytest.raises(ValueError, match="not finite"):
        envelope_of(contexts, missing, QUANTILES)


def test_chunks_written_are_kept_and_the_rest_answered(tmp_path: Path) -> None:
    contexts = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    handed: list[int] = []

    def counting(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        handed.append(len(batch))
        return spread_around_last(batch)

    first = run(contexts, tmp_path, counting, QUANTILES, chunk=2)
    assert handed == [2, 2, 1]
    (tmp_path / "chunk-0000002.npy").unlink()

    again = run(contexts, tmp_path, counting, QUANTILES, chunk=2)

    assert handed == [2, 2, 1, 2]
    assert np.array_equal(first, again)
    assert again[4, 1].tolist() == pytest.approx([29.1, 29.5, 29.9])


def test_chunks_of_other_contexts_are_refused(tmp_path: Path) -> None:
    contexts = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    run(contexts, tmp_path, spread_around_last, QUANTILES, chunk=2)

    with pytest.raises(SystemExit, match="chunks of other contexts"):
        run(contexts + 1.0, tmp_path, spread_around_last, QUANTILES, chunk=2)


def test_a_change_in_any_chunks_first_window_is_refused(tmp_path: Path) -> None:
    contexts = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    run(contexts, tmp_path, spread_around_last, QUANTILES, chunk=2)
    changed = contexts.copy()
    changed[2] += 1.0

    with pytest.raises(SystemExit, match="chunks of other contexts"):
        run(changed, tmp_path, spread_around_last, QUANTILES, chunk=2)
