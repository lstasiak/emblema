import torch
from torch import Tensor


class DeviceGenerator:
    """The generator a device draws from, saved and put back with the rest of a run's state.

    Dropout on an accelerator does not draw from the host's generator, so a run resumed with the
    host's state alone would drop different units than the run it continues. Which call reaches
    that state differs per backend and the host keeps none of its own beyond the global one, so
    the device is held here once and the difference stays in one place.
    """

    def __init__(self, device: str) -> None:
        self._kind = torch.device(device).type

    def state(self) -> Tensor | None:
        """What this device's generator holds, or ``None`` where it keeps nothing of its own."""
        if self._kind == "cuda":  # pragma: no cover - exercised on the GPU platforms, never here
            return torch.cuda.get_rng_state()
        if self._kind == "mps":
            return torch.mps.get_rng_state()
        return None

    def restore(self, state: Tensor | None) -> None:
        """Put back what ``state`` took, where there was anything to take."""
        if state is None:
            return
        if self._kind == "cuda":  # pragma: no cover - exercised on the GPU platforms, never here
            torch.cuda.set_rng_state(state)
        elif self._kind == "mps":
            torch.mps.set_rng_state(state)
