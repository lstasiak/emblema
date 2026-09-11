"""The accelerator leg of the loader question, skipped everywhere the accelerator is absent.

Pretraining runs on MPS on the development machine, and a loader that cannot fill the accelerator
turns a GPU budget into a CPU budget. The assertion is a ratio rather than a duration, because a
duration would only say which machine ran the test. The model is the stand-in encoder of the export
suite at the width and depth of compute tier M: the real encoder arrives later, and what a step
costs follows from its shape.

The margin and the way it is measured come from the report script, so the number this asserts and
the number that script prints cannot drift apart.
"""

import pytest
import torch

from scripts.loader_throughput_report import (
    MARGIN,
    budget,
    collate_seconds,
    default_window,
    step_seconds,
    synthetic_windows,
    transfer_seconds,
)

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS"),
]

TIER = next(tier for tier in budget()["tiers"] if tier["name"] == "M")
WINDOWS = 64


def test_a_training_step_costs_more_than_putting_the_batch_in_front_of_it() -> None:
    windows = list(synthetic_windows(WINDOWS, default_window()))

    # Without workers a step pays for the batch it consumes, collating it and then moving it
    # across; both belong on the feeding side of the comparison.
    feeding = collate_seconds(windows, num_workers=0) + transfer_seconds(windows, device="mps")
    step = step_seconds(windows, width=TIER["width"], layers=TIER["layers"], device="mps")

    assert step > MARGIN * feeding, (
        f"a step took {step * 1e3:.0f} ms, feeding it {feeding * 1e3:.0f} ms"
    )
