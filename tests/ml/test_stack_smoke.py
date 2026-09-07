"""ML stack smoke: torch and onnxruntime wheels exist on this interpreter.

Imports are hard on purpose: a missing wheel must fail the CI matrix entry, not skip it.
"""

import sys

import pytest

pytestmark = pytest.mark.ml


def test_python_meets_project_floor() -> None:
    assert sys.version_info >= (3, 12)


def test_torch_runs_on_cpu() -> None:
    import torch

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert torch.allclose(x.sum(dim=1), torch.tensor([3.0, 12.0]))


def test_onnxruntime_has_cpu_provider() -> None:
    import onnxruntime as ort

    assert "CPUExecutionProvider" in ort.get_available_providers()
