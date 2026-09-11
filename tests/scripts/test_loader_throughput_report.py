"""What the throughput report measures is machine-dependent; what it measures it *on* is not.

The ratio the report prints can only be checked where an accelerator exists, so its assertion lives
in an MPS-only test that CI never runs. The corpus that test feeds is built here, on any machine: a
generator that raised would otherwise take the DoD assertion down with it, silently, on the one
machine that does run it.
"""

import pytest

pytest.importorskip("torch")

from emblema.shared.adapters.loaders.window_loader import WindowLoader
from scripts.loader_throughput_report import (
    BATCH_SIZE,
    channel_count,
    default_window,
    order_seconds,
    synthetic_windows,
)

pytestmark = pytest.mark.ml


def test_the_stand_in_corpus_has_the_shape_of_the_default_window() -> None:
    window = default_window()

    windows = list(synthetic_windows(3, window))

    assert len(windows) == 3
    assert {len(one) for one in windows} == {int(window.length) * channel_count()}


def test_the_stand_in_corpus_reaches_the_loader_as_a_batch() -> None:
    windows = list(synthetic_windows(BATCH_SIZE, default_window()))

    loader = WindowLoader(windows, batch_size=BATCH_SIZE, seed=1)

    assert next(loader.batches_of(0)).batch_size == BATCH_SIZE


def test_ordering_is_timed_over_a_fresh_epoch_every_repeat() -> None:
    # Every repeat advances the epoch, so a bounded supply of them would end the run rather than
    # the measurement.
    assert order_seconds(8) > 0.0
