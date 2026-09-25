from dataclasses import dataclass, replace
from typing import ClassVar, Self

from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.classical.ridge_spec import RidgeSpec
from emblema.evaluation.domain.exceptions import UnknownKnobError
from emblema.evaluation.domain.tuning.knob import turned


@dataclass(frozen=True, kw_only=True)
class RandomConvolutions:
    """MiniRocket: a window laid on a grid, read by random convolutions, answered by ridge.

    Strong exactly where a pretrained encoder hopes to win — a few hundred labels — because the
    convolutions are not learnt and the only thing fitted is a linear map, which has little
    variance to spend. Its price is the grid: the channels are an axis of the input, so a fit
    stays in the corpus whose layout it was made for, and a corpus observed at irregular instants
    is read after it has been resampled, which is a loss this method cannot avoid.

    Attributes:
        convolutions: How many features the convolutions make, and over how fine a grid.
        ridge: What the linear map on top of them chooses its penalty among.
    """

    convolutions: MiniRocketSpec
    ridge: RidgeSpec

    NAME: ClassVar[str] = "random_convolutions"
    # What a selection may turn, under the names the recipe records them by. The ridge penalty
    # is chosen inside every fit already, by leave-one-out error, so it is not turned here.
    KNOBS: ClassVar[dict[str, str]] = {
        "convolution_features": "features",
        "grid_resolution": "grid_resolution",
    }

    @property
    def spans_channel_layouts(self) -> bool:
        """Never: the channels are an axis of what is convolved."""
        return False

    def parameters(self) -> dict[str, str | int | float]:
        """The method flattened to scalars, in a fixed order, for whoever records a fit."""
        stated: dict[str, str | int | float] = {"method": self.NAME}
        return stated | self.convolutions.parameters() | self.ridge.parameters()

    def tuned(self, knob: str, value: str) -> Self:
        """These convolutions with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the method has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"{self} has no knob {knob!r}; it turns {tuple(self.KNOBS)}")
        return replace(self, convolutions=turned(self.convolutions, self.KNOBS[knob], value))

    def __str__(self) -> str:
        return self.NAME
