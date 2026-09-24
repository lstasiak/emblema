import math

import numpy as np
import pytest

from emblema.evaluation.adapters.features.spectral_features import SpectralFeatures
from emblema.evaluation.adapters.features.window_spectrum import WindowSpectrum
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from tests.evaluation.adapters.features.support import timed, window

TIMES = np.arange(1, 65) / 64
WAVE = timed(2, [(float(np.sin(2 * np.pi * 6 * t)), float(t)) for t in TIMES])


def test_each_channel_of_the_corpus_has_a_block_in_vocabulary_order() -> None:
    features = SpectralFeatures(3)

    assert features.width == 3 * len(WindowSpectrum.NAMES) == len(features.names())
    assert features.names()[len(WindowSpectrum.NAMES)] == "channel_2_spectrum_band_1_2"


def test_a_channel_the_window_does_not_hold_leaves_its_block_absent() -> None:
    row = SpectralFeatures(3).of([window(WAVE)])[0]
    width = len(WindowSpectrum.NAMES)

    assert all(math.isnan(value) for value in row[:width])
    assert row[width + WindowSpectrum.NAMES.index("band_4_8")] > 0.5
    assert all(math.isnan(value) for value in row[2 * width :])


def test_a_channel_beyond_the_vocabulary_is_refused() -> None:
    with pytest.raises(UnreadableTaskCorpusError, match="holds channel 2"):
        SpectralFeatures(1).of([window(WAVE)])


def test_a_corpus_of_no_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        SpectralFeatures(0)
