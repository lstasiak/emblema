"""Why real gradient-boosted trees and torch cannot both work in one test process here.

The XGBoost wheel for this platform ships no OpenMP runtime of its own: its library records a
dependency on ``@rpath/libomp.dylib`` and an rpath into Homebrew's prefix, so the runtime it
fits with is the one installed system-wide. Torch ships its own copy and loads it by its own
path. Both are LLVM's, and both export the template instantiations of ``__kmp_suspend_64`` as
weak definitions, which the loader coalesces across the whole process: whichever copy was loaded
first answers for both, so a thread started by one runtime ends up in the other's code and on
the other's state. Either order takes the interpreter down, not a test with it: XGBoost loaded
first crashes torch's first parallel kernel, torch loaded first crashes the fit.

Nothing installed wrongly causes this and no reinstallation fixes it; scikit-learn escapes it by
carrying its own copy, and loading torch's copy ahead does not help, because the loader keeps the
two files apart. It is also not a constraint the application has to live with, because no
process of it holds both: the worker that adapts a backbone has torch, the worker that fits
classical candidates has neither torch nor any use for it. The test process is the only place
that puts them together, so it draws the same boundary in one of two ways. By default torch is
loaded before XGBoost, whatever tests are chosen and in whatever order they are collected, and
the fits are skipped; with ``--without-torch`` torch is refused, and the fits run. The image the
suite runs in has one OpenMP runtime for everyone, so there the fits run in the one process.
"""

import importlib
import sys
from collections.abc import Sequence
from contextlib import suppress
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

WITHOUT_TORCH = "--without-torch"


class TorchBeforeXgboost(MetaPathFinder):
    """Loads torch, where it is installed, the moment XGBoost is first imported.

    Torch first leaves the process able to run torch and only the fits unsafe, which the skip
    below catches; XGBoost first leaves torch unsafe, which no skip can catch, since any test
    that runs a network would crash.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == "xgboost" and "torch" not in sys.modules:
            with suppress(ModuleNotFoundError):
                importlib.import_module("torch")
        return None


class TorchRefused(MetaPathFinder):
    """Refuses torch, so that a run of the fits cannot be taken down by a test that loads it."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"torch is kept out of a run {WITHOUT_TORCH}", name=fullname)
        return None


def draw_the_boundary(*, without_torch: bool) -> None:
    """Put the finder for this run ahead of every other, before any test module is imported."""
    if without_torch:
        if "torch" in sys.modules:
            raise pytest.UsageError(f"torch was loaded before {WITHOUT_TORCH} could refuse it")
        sys.meta_path.insert(0, TorchRefused())
    elif sys.platform == "darwin":
        sys.meta_path.insert(0, TorchBeforeXgboost())


def skip_if_torch_shares_the_process() -> None:
    """Skip when fitting here would take the interpreter down instead of answering."""
    if sys.platform == "darwin" and "torch" in sys.modules:
        pytest.skip(
            "torch is loaded in this process and its OpenMP runtime would be used for the fit; "
            f"run these {WITHOUT_TORCH} or in the image"
        )
