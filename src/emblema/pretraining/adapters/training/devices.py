"""Which device this process trains on where nothing says otherwise.

The experiment declares precision and scale, never hardware: the same configuration is meant to
run on a laptop's accelerator and on a rented one, and the tier it declares says which of those it
was measured on. A module rather than a class, because this is one fact about the machine and no
state: what a device's generator holds is ``DeviceGenerator``'s.
"""

import torch


def available_device() -> str:
    """The accelerator this machine has, the CPU where it has none."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
