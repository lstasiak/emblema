from dataclasses import dataclass
from math import ceil, isfinite
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidMiniRocketSpecError


@dataclass(frozen=True, kw_only=True)
class MiniRocketSpec:
    """How many random convolutions read a window.

    MiniRocket convolves a fixed family of 84 kernels, each at several dilations and against
    several biases, and every pairing yields one feature. A count that is not a whole multiple
    of the family would leave some kernels with more features than others for no reason
    anybody chose, so the count is stated in whole multiples and refused otherwise.

    The grid a window is laid on first is counted in the corpus's own time — steps per cycle of
    an engine, per hour of a stay — so one setting means the same thing for every corpus, where
    a count of steps set once would fit one window length and distort every other. At one step
    per unit, the default, a corpus sampled on that cadence is read as it was recorded, which
    is how the method was published; any other resolution is a knob chosen by a declared
    selection, never set by hand.

    Invariants: at least one feature per kernel, in whole multiples of the family; the
    resolution is positive and finite.

    Attributes:
        features: Features the convolutions produce, a whole multiple of 84.
        grid_resolution: Steps of the grid per unit of the corpus's time.
    """

    features: int
    grid_resolution: float = 1.0

    KERNELS: ClassVar[int] = 84
    KERNEL_LENGTH: ClassVar[int] = 9

    def __post_init__(self) -> None:
        if self.features < self.KERNELS or self.features % self.KERNELS:
            raise InvalidMiniRocketSpecError(
                f"features must be a positive multiple of {self.KERNELS}, got {self.features}"
            )
        if not isfinite(self.grid_resolution) or self.grid_resolution <= 0.0:
            raise InvalidMiniRocketSpecError(
                f"grid_resolution must be positive and finite, got {self.grid_resolution}"
            )

    def steps_over(self, window_length: float) -> int:
        """How many steps a window of ``window_length`` units of time is laid on."""
        return ceil(window_length * self.grid_resolution)

    def parameters(self) -> dict[str, int | float]:
        """The knobs flattened to scalars, in a fixed order, for whoever records a fit."""
        return {"convolution_features": self.features, "grid_resolution": self.grid_resolution}
