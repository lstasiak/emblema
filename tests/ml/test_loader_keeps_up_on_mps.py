"""The accelerator leg of the loader question, skipped everywhere the accelerator is absent.

Pretraining runs on MPS on the development machine, and a loader that cannot fill the accelerator
turns a GPU budget into a CPU budget. The assertion is a ratio rather than a duration, because a
duration would only say which machine ran the test. The model is the encoder at the shape of
compute tier M.

The margin and the way it is measured come from the report script, so the number this asserts and
the number that script prints cannot drift apart.
"""

import pytest
import torch

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.shared.kernel.compute import ComputeTier
from scripts.loader_throughput_report import (
    MARGIN,
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

TIER = architecture_of(ComputeTiers.load().profile(ComputeTier.M))
WINDOWS = 64


def test_a_training_step_costs_more_than_putting_the_batch_in_front_of_it() -> None:
    windows = list(synthetic_windows(WINDOWS, default_window()))

    # Without workers a step pays for the batch it consumes, collating it and then moving it
    # across; both belong on the feeding side of the comparison.
    feeding = collate_seconds(windows, num_workers=0) + transfer_seconds(windows, device="mps")
    step = step_seconds(windows, architecture=TIER, device="mps")

    assert step > MARGIN * feeding, (
        f"a step took {step * 1e3:.0f} ms, feeding it {feeding * 1e3:.0f} ms"
    )
