from dataclasses import dataclass
from math import isfinite, log
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidCandidateMethodError


@dataclass(frozen=True, kw_only=True)
class MethodParameter:
    """One knob of how a candidate learns, under the name its provider gives it.

    Invariants: the name is non-blank without surrounding whitespace.

    Attributes:
        name: What the provider calls the knob.
        value: What it was set to, as the provider spells it.
    """

    name: str
    value: str

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidCandidateMethodError(
                "a parameter must be named, non-blank without surrounding whitespace"
            )


@dataclass(frozen=True)
class CandidateMethod:
    """How a candidate learns, as the provider that supplies it states it.

    Names and text, because a campaign cannot interpret them and must not have to (ADR-0035):
    recorded verbatim, read by a person. Without them a stored campaign says how much arithmetic
    its arms were allowed and nothing about what they did with it.

    Name order and one spelling per value are what make two runs of the same arm recognisably
    the same, including after a round trip through a store that keeps no order of its own.

    Invariants: no parameter is named twice; the parameters are in name order.

    Attributes:
        parameters: The knobs, in name order.
    """

    parameters: tuple[MethodParameter, ...] = ()

    def __post_init__(self) -> None:
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise InvalidCandidateMethodError(f"a parameter is named twice: {sorted(names)}")
        if names != sorted(names):
            raise InvalidCandidateMethodError(f"parameters must be in name order: {names}")

    @classmethod
    def of(cls, **parameters: object) -> Self:
        """The method these keyword arguments describe, each value written as its text.

        The one way this is built, so that whoever states a method need not state it in order
        and whoever reads one back gets the order it was canonicalised into.
        """
        return cls(
            tuple(
                MethodParameter(name=name, value=str(value))
                for name, value in sorted(parameters.items())
            )
        )

    def departure_from(self, default: "CandidateMethod") -> tuple[int, float]:
        """How far this method departs from ``default``: how many knobs differ, then how far.

        A knob with a positive number on both sides departs by the size of the ratio between
        them on a log scale, so doubling and halving are as far as each other; any other knob
        that differs departs by one. What a selection breaks ties towards is the smaller of two
        departures, which is what makes it prefer the setting a method was published with.
        """
        stated = {parameter.name: parameter.value for parameter in default.parameters}
        changed, distance = 0, 0.0
        for parameter in self.parameters:
            before = stated.get(parameter.name)
            if before == parameter.value:
                continue
            changed += 1
            distance += self._ratio(before, parameter.value)
        return changed, distance

    @staticmethod
    def _ratio(before: str | None, after: str) -> float:
        try:
            was, now = float(before or ""), float(after)
        except ValueError:
            return 1.0
        if not (isfinite(was) and isfinite(now)) or was <= 0.0 or now <= 0.0:
            return 1.0
        return abs(log(now / was))
