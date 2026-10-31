from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidRateChoiceError


@dataclass(frozen=True, kw_only=True)
class RateOption:
    """One rate or setting of an arm beside the rate in force: its gain and that gain's error.

    A setting is what a candidate states beside its rate that the candidate in force does not, as
    fine-tuning with the encoder learning at a share of the rate and its head solved first; such
    an option runs at the rate it states and is read by the same rule as a rate.

    Invariants: the rate is positive and finite, the gain finite, the error finite and not
    negative.

    Attributes:
        rate: The learning rate.
        setting: What else the option states beside its rate; empty for a rate alone.
        gain: Its mean area over every repeat of the arm less the rate in force's.
        standard_error: Standard error of that gain over resamples of the units.
    """

    rate: float
    gain: float
    standard_error: float
    setting: str = ""

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
    """The rate an arm keeps: the rate in force, unless another rate or setting clearly does better.

    An option clears the bar where its gain exceeds one standard error of that gain; of several
    that clear it, the largest gain is kept. A rate chosen at either end of the grid of rates is
    named, since the best rate may lie beyond it; a setting has no end to lie at.

    Invariants: no rate and setting is named twice, the rate in force with no setting among them;
    the rate in force is positive and finite.

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
        named = [(self.in_force, ""), *((option.rate, option.setting) for option in self.others)]
        if len(set(named)) != len(named):
            raise InvalidRateChoiceError(f"a grid names a rate and setting twice: {named}")

    @property
    def chosen_option(self) -> RateOption | None:
        """The option kept; ``None`` where the rate in force stays."""
        cleared = [option for option in self.others if option.clears]
        if not cleared:
            return None
        return max(cleared, key=lambda option: option.gain)

    @property
    def chosen(self) -> float:
        """The rate kept, whatever setting comes with it."""
        option = self.chosen_option
        return self.in_force if option is None else option.rate

    @property
    def at_edge(self) -> bool:
        option = self.chosen_option
        if option is not None and option.setting:
            return False
        rates = [self.in_force, *(each.rate for each in self.others if not each.setting)]
        return len(rates) > 1 and self.chosen in (min(rates), max(rates))
