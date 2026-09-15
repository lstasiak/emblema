from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

import torch

from emblema.pretraining.domain.exceptions import UnsupportedPrecisionError
from emblema.pretraining.domain.training.precision import Precision


class TorchPrecision:
    """What a device does to honour the precision an experiment declared, or why it cannot.

    Each backend supports a different set, and the difference is not a detail of speed: half
    precision on one is a fused kernel and on another a silent fall back to single. So the pairs
    are written out and an unsupported one is refused, rather than demoted to what the machine
    happens to have. Gradient scaling follows the same rule — it is CUDA's answer to half
    precision's underflow, and where it is unavailable the run says so by not scaling rather than
    by pretending to.

    Attributes:
        dtype: What autocast computes in, or ``None`` where the run stays in single precision.
    """

    # Refusals, each for a reason: bfloat16 on MPS is not implemented by the backend, and half
    # precision on CPU runs without the hardware that makes it worth having.
    _SUPPORTED: ClassVar[dict[str, tuple[Precision, ...]]] = {
        "cpu": (Precision.FP32, Precision.BF16),
        "mps": (Precision.FP32, Precision.FP16),
        "cuda": (Precision.FP32, Precision.FP16, Precision.BF16),
    }
    _DTYPES: ClassVar[dict[Precision, torch.dtype]] = {
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }

    def __init__(self, precision: Precision, device: str) -> None:
        """Honour ``precision`` on ``device``.

        Raises:
            UnsupportedPrecisionError: If the device cannot compute at that precision.
        """
        self._kind = torch.device(device).type
        if precision not in self._SUPPORTED.get(self._kind, ()):
            raise UnsupportedPrecisionError(
                f"{self._kind} cannot train at {precision}; it runs "
                f"{', '.join(self._SUPPORTED.get(self._kind, ()))}"
            )
        self._precision = precision

    @property
    def dtype(self) -> torch.dtype | None:
        return self._DTYPES.get(self._precision)

    @property
    def scales_gradients(self) -> bool:
        """Whether the gradients are scaled before the backward pass."""
        return self._precision is Precision.FP16 and self._kind == "cuda"

    def autocast(self) -> AbstractContextManager[None]:
        """The context the forward pass and the loss are computed in."""
        dtype = self.dtype
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self._kind, dtype=dtype)

    def scaler(self) -> torch.amp.GradScaler:
        """The gradient scaler for this precision; a scaler that does nothing where none is due.

        A disabled scaler passes the loss, the step and the update straight through, so the
        training loop has one shape whatever the precision.
        """
        return torch.amp.GradScaler(self._kind, enabled=self.scales_gradients)
