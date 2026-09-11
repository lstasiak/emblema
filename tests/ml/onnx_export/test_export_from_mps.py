"""The Apple-silicon leg of the export question, skipped everywhere the accelerator is absent.

Pretraining runs on MPS on the development machine, so the weights that will one day be exported
were produced there. Exporting traces CPU tensors, which leaves one thing worth asserting rather
than assuming: moving a model back to the CPU and exporting it reproduces what it computed on the
accelerator, within the tolerance a different reduction order costs.
"""

import numpy as np
import pytest
import torch

from tests.ml.onnx_export.batches import random_batch
from tests.ml.onnx_export.dummy_set_encoder import DummySetEncoder
from tests.ml.onnx_export.exported_encoder import ExportedEncoder

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS"),
    pytest.mark.filterwarnings("ignore:# The axis name:UserWarning"),
]

# Looser than the CPU-to-ONNX comparison: MPS accumulates the same float32 arithmetic in a
# different order, so the two sides may differ by more than a round trip between kernels does.
ATOL = 1e-4
RTOL = 1e-3


def test_a_model_that_ran_on_mps_exports_once_moved_back_to_the_cpu() -> None:
    torch.manual_seed(1)
    model = DummySetEncoder().eval()
    batch = random_batch(1, 512, seed=512)
    with torch.no_grad():
        on_accelerator = model.to("mps")(*(tensor.to("mps") for tensor in batch.args))

    exported = ExportedEncoder.from_model(model.to("cpu"))

    np.testing.assert_allclose(
        exported.run_onnx(batch), on_accelerator.cpu().numpy(), rtol=RTOL, atol=ATOL
    )
