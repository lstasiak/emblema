"""The accelerator leg of the loader question, skipped everywhere the accelerator is absent.

Pretraining runs on MPS on the development machine, and a loader that cannot fill the accelerator
turns a GPU budget into a CPU budget. The assertion is a ratio rather than a duration, because a
duration would only say which machine ran the test. The model is the stand-in encoder of the export
suite at the width and depth of compute tier M: the real encoder arrives later, and what a step
costs follows from its shape.
"""

import pytest
import torch

from scripts.loader_throughput_report import (
    budget,
    collate_seconds,
    default_window,
    step_seconds,
    synthetic_windows,
)

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS"),
]

TIER = next(tier for tier in budget()["tiers"] if tier["name"] == "M")
WINDOWS = 64
# Collating happens while the previous step runs only with workers; with none, a step must pay for
# the batch it consumes. Twice its cost is a floor, not a target — it was three orders of magnitude
# when this was written, and a fall to single digits is the signal to move the work off the path.
MARGIN = 2.0


def test_a_training_step_costs_more_than_collating_the_batch_it_consumes() -> None:
    windows = list(synthetic_windows(WINDOWS, default_window()))

    collating = collate_seconds(windows, num_workers=0)
    step = step_seconds(windows, width=TIER["width"], layers=TIER["layers"], device="mps")

    assert step > MARGIN * collating, (
        f"a step took {step * 1e3:.0f} ms, collating its batch {collating * 1e3:.0f} ms"
    )
