from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError


@dataclass(frozen=True)
class ForecastScheme:
    """What a window is asked to say about one sensor: its exact reading a fixed time past the end.

    A forecast of the reading rather than the reading itself, because the reading at the window's
    end is close to what a reconstruction objective already teaches, and a fixed horizon asks for
    what the window does not show: where the signal is going, which takes the frequencies behind
    it as well as the phase and amplitude the window reveals. The target is the reading exactly,
    without noise, in the sensor's own units: no ceiling and no transform, so an error is an error
    of the sensor.

    Invariants: the channel is non-blank without surrounding whitespace; the horizon is finite and
    not negative.

    Attributes:
        channel: Name of the sensor whose reading is asked for.
        horizon: How far past the window's end the reading is taken, in the unit's time axis;
            zero asks for the reading at the end itself.
    """

    channel: str
    horizon: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidLabelSchemeError(
                f"channel must be non-blank without surrounding whitespace: {self.channel!r}"
            )
        if not (isfinite(self.horizon) and self.horizon >= 0.0):
            raise InvalidLabelSchemeError(
                f"horizon must be finite and not negative: {self.horizon}"
            )

    @property
    def scale(self) -> float:
        """The unit targets are learnt in: the sensor's own, whose signal has unit variance."""
        return 1.0

    def instant_of(self, ends_at: float) -> float:
        """The moment the reading is taken for a window ending at ``ends_at``."""
        return ends_at + self.horizon

    def target(self, *, exact: float) -> float:
        """The target of a window whose exact reading at its instant is ``exact``: that reading."""
        return exact
