from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import InvalidCandidateVariantError


@dataclass(frozen=True, kw_only=True)
class CandidateVariant:
    """A candidate with some of its knobs turned, named in full by its reference.

    The name is the whole of the variant — ``minirocket@grid_resolution=1.28`` — so the process
    that declares a campaign and the one that runs its cells read the same variant out of the
    same text, with nothing configured on either side to keep in step. Knobs are named in name
    order, so one variant has one name however its knobs were written.

    Invariants: the base names no knob of its own; at least one knob is turned when any is
    named, none twice; knob names and values are non-blank and hold no separator.

    Attributes:
        base: The candidate whose knobs are turned.
        knobs: Each knob turned and the value it is set to, as text, in name order.
    """

    base: CandidateRef
    knobs: tuple[tuple[str, str], ...]

    AT: ClassVar[str] = "@"
    BETWEEN: ClassVar[str] = ","
    EQUALS: ClassVar[str] = "="

    def __post_init__(self) -> None:
        if self.AT in str(self.base):
            raise InvalidCandidateVariantError(f"a base candidate names no knob: {self.base}")
        names = [name for name, _ in self.knobs]
        if len(set(names)) != len(names):
            raise InvalidCandidateVariantError(f"a knob is turned twice: {sorted(names)}")
        if names != sorted(names):
            raise InvalidCandidateVariantError(f"knobs must be in name order: {names}")
        for name, value in self.knobs:
            for text in (name, value):
                if (
                    not text
                    or text != text.strip()
                    or any(mark in text for mark in (self.AT, self.BETWEEN, self.EQUALS))
                ):
                    raise InvalidCandidateVariantError(f"not a knob and its value: {name}={value}")

    @classmethod
    def parse(cls, ref: CandidateRef) -> Self:
        """The variant ``ref`` names; a reference with no knob is its own base.

        Raises:
            InvalidCandidateVariantError: If the text is not a base and its knobs.
        """
        base, at, turned = str(ref).partition(cls.AT)
        if not at:
            return cls(base=ref, knobs=())
        if not turned:
            raise InvalidCandidateVariantError(f"{ref} names no knob after {cls.AT!r}")
        knobs = []
        for pair in turned.split(cls.BETWEEN):
            name, equals, value = pair.partition(cls.EQUALS)
            if not equals:
                raise InvalidCandidateVariantError(f"{pair!r} in {ref} is not knob=value")
            knobs.append((name, value))
        return cls(base=CandidateRef(base), knobs=tuple(knobs))

    def applied_to[T](self, default: T, turn: Callable[[T, str, str], T]) -> T:
        """``default`` with every knob of this variant turned by ``turn``, knob by knob.

        A variant that turns knobs and lands on the default is refused: two names for one
        candidate would let a selection weigh the default against itself.

        Raises:
            UnknownKnobError: If ``turn`` refuses a knob or its value.
            InvalidCandidateVariantError: If the knobs turned leave the default as it was.
        """
        tuned = default
        for name, value in self.knobs:
            tuned = turn(tuned, name, value)
        if self.knobs and tuned == default:
            raise InvalidCandidateVariantError(f"{self.ref} turns no knob away from {self.base}")
        return tuned

    @property
    def ref(self) -> CandidateRef:
        """The name the variant competes under."""
        if not self.knobs:
            return self.base
        turned = self.BETWEEN.join(f"{name}{self.EQUALS}{value}" for name, value in self.knobs)
        return CandidateRef(f"{self.base}{self.AT}{turned}")
