from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidRateChoiceError


@dataclass(frozen=True, kw_only=True)
class RateOption:
    """One rate of an arm beside the rate in force: its gain and that gain's standard error.

    Invariants: the rate is positive and finite, the gain finite, the error finite and not
    negative.

    Attributes:
        rate: The learning rate.
        gain: Its mean area over every repeat of the arm less the rate in force's.
        standard_error: Standard error of that gain over resamples of the units.
    """

    rate: float
    gain: float
    standard_error: float

    def __post_init__(self) -> None:
        if not isfinite(self.rate) or self.rate <= 0.0:
            raise InvalidRateChoiceError(f"a rate must be positive and finite, got {self.rate}")
        if not isfinite(self.gain):
            raise InvalidRateChoiceError(f"a gain must be finite, got {self.gain}")
        if not isfinite(self.standard_error) or self.standard_error < 0.0:
            raise InvalidRateChoiceError(
                f"a standard error must be finite and not negative, got {self.standard_error}"
            )

    @property
    def clears(self) -> bool:
        """Whether its gain over the rate in force exceeds one standard error of that gain."""
        return self.gain > self.standard_error


@dataclass(frozen=True, kw_only=True)
class RateChoice:
    """The rate an arm keeps: the rate in force, unless another clearly does better.

    A rate clears the bar where its gain exceeds one standard error of that gain; of several
    that clear it, the largest gain is kept. A choice at either end of the grid is named, since
    the best rate may lie beyond it.

    Invariants: the rates are distinct and the rate in force is positive and finite.

    Attributes:
        in_force: The rate the arm runs at unless another is chosen.
        others: The other rates of the grid, each beside the rate in force.
    """

    in_force: float
    others: tuple[RateOption, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.in_force) or self.in_force <= 0.0:
            raise InvalidRateChoiceError(
                f"the rate in force must be positive and finite, got {self.in_force}"
            )
        rates = [self.in_force, *(option.rate for option in self.others)]
        if len(set(rates)) != len(rates):
            raise InvalidRateChoiceError(f"a grid of rates names a rate twice: {rates}")

    @property
    def chosen(self) -> float:
        cleared = [option for option in self.others if option.clears]
        if not cleared:
            return self.in_force
        return max(cleared, key=lambda option: option.gain).rate

    @property
    def at_edge(self) -> bool:
        rates = [self.in_force, *(option.rate for option in self.others)]
        return len(rates) > 1 and self.chosen in (min(rates), max(rates))
