"""The Apple-silicon leg of the export, skipped everywhere the accelerator is absent.

Candidates are fitted on MPS on the development machine, so the weights that are exported were
produced there. The exporter refuses a model held on the accelerator and the adapter moves it to
the host first, which leaves one thing worth asserting rather than assuming: the graph reproduces
what the candidate computed on the accelerator, within the tolerance a different reduction order
costs.
"""

import numpy as np
import pytest
import torch

from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from tests.evaluation.adapters.onnx.candidates import TARGET_SCALE, adapted
from tests.support.token_tensors import random_batch

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS"),
    pytest.mark.filterwarnings("ignore:# The axis name:UserWarning"),
]

# Looser than the host-to-graph comparison: MPS accumulates the same float32 arithmetic in a
# different order, so the two sides may differ by more than a round trip between kernels does.
ATOL = 1e-4
RTOL = 1e-3


def test_a_candidate_that_ran_on_mps_exports_once_moved_back_to_the_host() -> None:
    candidate = adapted(TransferMode.FULL_FINE_TUNING).to("mps")
    batch = random_batch(1, 512, seed=512)
    with torch.no_grad():
        on_accelerator = candidate(batch.to("mps")) * TARGET_SCALE

    graph = InferenceGraph.exported(candidate, target_scale=TARGET_SCALE)

    np.testing.assert_allclose(
        graph.predict(batch), on_accelerator.cpu().numpy(), rtol=RTOL, atol=ATOL * TARGET_SCALE
    )
