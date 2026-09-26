from dataclasses import dataclass
from math import ceil, isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidPatchModelSpecError, UnknownKnobError
from emblema.evaluation.domain.tuning.knob import turned


@dataclass(frozen=True, kw_only=True)
class PatchModelSpec:
    """How a patch model reads a window laid on a grid, and how large it is.

    Every channel's row of the grid is cut into overlapping patches of equal length, and each
    patch becomes one token of a transformer that is shared by all channels and sees one channel
    at a time. The window's end is padded by one stride, so the last reading starts a patch of
    its own and is not left to the tail of the one before it.

    The grid is counted in the corpus's own time, as it is for the convolutions: one setting
    means the same thing for every corpus, and at one step per unit a corpus sampled on that
    cadence is read as it was recorded.

    Invariants: the patch, the stride, the width, the heads, the layers and the feed-forward
    width are positive; the stride is no longer than the patch; the width divides into the
    heads; the dropout lies in ``[0, 1)``; the resolution is positive and finite.

    Attributes:
        patch_length: Steps of the grid one token covers.
        stride: Steps between the starts of two neighbouring patches.
        width: Size of a token's state.
        heads: Attention heads per layer.
        layers: Transformer layers.
        feedforward_width: Hidden width of each layer's feed-forward network.
        dropout: Share of activations dropped while the model learns.
        grid_resolution: Steps of the grid per unit of the corpus's time.
    """

    patch_length: int
    stride: int
    width: int
    heads: int
    layers: int
    feedforward_width: int
    dropout: float
    grid_resolution: float

    # Every field is a knob: none of them touches the compute budget, which is the schedule's,
    # and a selection that may turn the rate but not the depth would be tuning half a network.
    KNOBS: ClassVar[tuple[str, ...]] = (
        "patch_length",
        "stride",
        "width",
        "heads",
        "layers",
        "feedforward_width",
        "dropout",
        "grid_resolution",
    )

    def __post_init__(self) -> None:
        for label, count in (
            ("patch_length", self.patch_length),
            ("stride", self.stride),
            ("width", self.width),
            ("heads", self.heads),
            ("layers", self.layers),
            ("feedforward_width", self.feedforward_width),
        ):
            if count < 1:
                raise InvalidPatchModelSpecError(f"{label} must be positive, got {count}")
        if self.stride > self.patch_length:
            raise InvalidPatchModelSpecError(
                f"a stride of {self.stride} would skip steps between patches of {self.patch_length}"
            )
        if self.width % self.heads:
            raise InvalidPatchModelSpecError(
                f"a width of {self.width} does not divide into {self.heads} heads"
            )
        if not isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise InvalidPatchModelSpecError(f"dropout must lie in [0, 1), got {self.dropout}")
        if not isfinite(self.grid_resolution) or self.grid_resolution <= 0.0:
            raise InvalidPatchModelSpecError(
                f"grid_resolution must be positive and finite, got {self.grid_resolution}"
            )

    def tuned(self, knob: str, value: str) -> Self:
        """This shape with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the shape has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"a patch model has no knob {knob!r}; it turns {self.KNOBS}")
        return turned(self, knob, value)

    def steps_over(self, window_length: float) -> int:
        """How many steps a window of ``window_length`` units of time is laid on."""
        return ceil(window_length * self.grid_resolution)

    def patches_over(self, steps: int) -> int:
        """How many tokens a row of ``steps`` steps becomes, the padded end included.

        Raises:
            InvalidPatchModelSpecError: If the row is shorter than one patch.
        """
        if steps < self.patch_length:
            raise InvalidPatchModelSpecError(
                f"a row of {steps} steps is shorter than a patch of {self.patch_length}"
            )
        return (steps - self.patch_length) // self.stride + 2

    def parameters(self) -> dict[str, int | float]:
        """The shape flattened to scalars, in a fixed order, for whoever records a run."""
        return {
            "patch_length": self.patch_length,
            "stride": self.stride,
            "width": self.width,
            "heads": self.heads,
            "layers": self.layers,
            "feedforward_width": self.feedforward_width,
            "dropout": self.dropout,
            "grid_resolution": self.grid_resolution,
        }
