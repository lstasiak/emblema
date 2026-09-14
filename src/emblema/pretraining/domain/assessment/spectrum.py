from collections.abc import Sequence
from dataclasses import dataclass

# A frequency holding less than this share of the truth's energy is noise, not signal.
INFORMATIVE_SHARE = 0.01


@dataclass(frozen=True)
class Spectrum:
    """Energy per cycle per window, from one upwards, over the channels hidden whole."""

    truth: tuple[float, ...]
    model_residual: tuple[float, ...]
    ridge_residual: tuple[float, ...]
    fitted: int
    skipped: int

    def shares(self) -> tuple[float, ...]:
        total = sum(self.truth)
        return tuple(energy / total if total > 0.0 else 0.0 for energy in self.truth)

    def informative(self) -> tuple[int, ...]:
        """Cycles per window at which the truth holds a share of its energy worth recovering."""
        return tuple(k + 1 for k, share in enumerate(self.shares()) if share >= INFORMATIVE_SHARE)

    def recovered(self, residual: Sequence[float]) -> tuple[float, ...]:
        """Per frequency, the share of the truth's energy the residual no longer holds."""
        return tuple(
            1.0 - left / truth if truth > 0.0 else 0.0
            for truth, left in zip(self.truth, residual, strict=True)
        )
