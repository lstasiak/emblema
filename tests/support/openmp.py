"""Why a fit of real gradient-boosted trees cannot run in a process that has loaded torch here.

The XGBoost wheel for this platform ships no OpenMP runtime of its own: its library records a
dependency on ``@rpath/libomp.dylib`` and an rpath into Homebrew's prefix, so the runtime it
fits with is the one installed system-wide. Torch ships its own copy. Both are LLVM's, and both
export the template instantiations of ``__kmp_suspend_64`` as weak definitions, which the loader
coalesces across the whole process: whichever copy was loaded first answers for both, so a
thread started by one runtime ends up in the other's code and on the other's state. The result
is a crash of the interpreter, not a failure a test could report.

Nothing installed wrongly causes this and no reinstallation fixes it; scikit-learn escapes it by
carrying its own copy under a private name. It is also not a constraint the application has to
live with, because no process of it holds both: the worker that adapts a backbone has torch, the
worker that fits classical candidates has neither torch nor any use for it. The test process was
the only place that put them together, and this skip is that same boundary drawn around it. The
image the suite runs in has one OpenMP runtime for everyone, so there the fits run.
"""

import sys

import pytest


def skip_if_torch_shares_the_process() -> None:
    """Skip when fitting here would take the interpreter down instead of answering."""
    if sys.platform == "darwin" and "torch" in sys.modules:
        pytest.skip(
            "torch is loaded in this process and its OpenMP runtime would be used for the fit; "
            "run these alone or in the image"
        )
