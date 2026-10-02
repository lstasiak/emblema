"""The test process keeps XGBoost's OpenMP runtime from taking torch down, in either mode.

The crash this guards against happens at the interpreter's level, so the order that used to
cause it is replayed in an interpreter of its own, which a crash would end with a signal rather
than with this test process.
"""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.support.openmp import TorchRefused

ROOT = Path(__file__).parents[2]

# XGBoost imported before torch, then a kernel torch spreads over threads: without the boundary
# the interpreter dies on that kernel.
XGBOOST_FIRST = """
from tests.support.openmp import draw_the_boundary
draw_the_boundary(without_torch={without_torch})
import xgboost
import torch
torch.set_num_threads(4)
layer = torch.nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
layer(torch.randn(8, 137, 64))
print("answered")
"""


def replay(*, without_torch: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", XGBOOST_FIRST.format(without_torch=without_torch)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.skipif(
    find_spec("torch") is None or find_spec("xgboost") is None,
    reason="needs both torch and XGBoost installed",
)
def test_torch_still_answers_after_xgboost_is_imported_first() -> None:
    run = replay(without_torch=False)

    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "answered"


@pytest.mark.skipif(find_spec("xgboost") is None, reason="needs XGBoost installed")
def test_without_torch_the_run_cannot_load_it() -> None:
    run = replay(without_torch=True)

    assert run.returncode == 1
    assert "ModuleNotFoundError: torch is kept out of a run --without-torch" in run.stderr


@pytest.mark.parametrize("name", ["torch", "torch.nn", "torch.nn.functional"])
def test_torch_and_its_modules_are_refused(name: str) -> None:
    with pytest.raises(ModuleNotFoundError, match="--without-torch"):
        TorchRefused().find_spec(name, None)


@pytest.mark.parametrize("name", ["numpy", "xgboost", "torchvision", "torchaudio.io"])
def test_anything_else_is_left_to_the_other_finders(name: str) -> None:
    assert TorchRefused().find_spec(name, None) is None
