from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidPairedUnitErrorsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.transfer.unit_error import UnitError


@dataclass(frozen=True, kw_only=True)
class PairedUnitErrors:
    """Two candidates' errors over the same units, unit by unit, so their difference is paired.

    The unit is what a comparison is paired on and what an interval resamples: the error of a
    control and of a candidate are held per unit in one order, and any group of units — the whole
    side, or a resample of it — scores each side by adding sums and counts before a root is
    taken. Repeats of a cell pool here as well: the squared errors of every repeat are added per
    unit, so a cell's error is the root of the mean squared error over its repeats, and the
    spread between repeats is reported beside the interval rather than folded into it.

    Invariants: at least one unit; both sides name the same units in the same order and count the
    same windows for each, since a pair over different windows is not a pair.

    Attributes:
        control: The error per unit of what the candidate is compared against.
        candidate: The error per unit of what is compared.
    """

    control: tuple[UnitError, ...]
    candidate: tuple[UnitError, ...]

    def __post_init__(self) -> None:
        if not self.control:
            raise InvalidPairedUnitErrorsError("a paired comparison needs at least one unit")
        if len(self.control) != len(self.candidate):
            raise InvalidPairedUnitErrorsError(
                f"{len(self.control)} control units paired with {len(self.candidate)} candidate "
                "units"
            )
        for left, right in zip(self.control, self.candidate, strict=True):
            if left.unit != right.unit:
                raise InvalidPairedUnitErrorsError(
                    f"control unit {left.unit} paired with candidate unit {right.unit}"
                )
            if left.windows != right.windows:
                raise InvalidPairedUnitErrorsError(
                    f"unit {left.unit} scored over {left.windows} control windows and "
                    f"{right.windows} candidate windows"
                )
        if len({error.unit for error in self.control}) != len(self.control):
            raise InvalidPairedUnitErrorsError("a unit is paired twice")

    @classmethod
    def pooled(
        cls,
        control: Sequence[Sequence[UnitError]],
        candidate: Sequence[Sequence[UnitError]],
    ) -> Self:
        """One pair out of the repeats of both sides, the errors added per unit across repeats.

        Each repeat is one run's error per unit; the repeats of a side are summed unit by unit,
        so a cell measured under several seeds is scored as one candidate over all of its answers.

        Raises:
            InvalidPairedUnitErrorsError: If a side has no repeat, its repeats disagree about
                the units, or the two sides do not pair.
        """
        return cls(control=_summed(control, "control"), candidate=_summed(candidate, "candidate"))

    @property
    def units(self) -> tuple[UnitKey, ...]:
        return tuple(error.unit for error in self.control)

    @property
    def rmse_control(self) -> float:
        return self.rmse_control_over(range(len(self.control)))

    @property
    def rmse_candidate(self) -> float:
        return self.rmse_candidate_over(range(len(self.candidate)))

    @property
    def reduction(self) -> float:
        """How much lower the candidate's error is than the control's, in the error's unit."""
        return self.rmse_control - self.rmse_candidate

    @property
    def relative_reduction(self) -> float:
        """The reduction as a share of the control's error.

        Raises:
            InvalidPairedUnitErrorsError: If the control makes no error, so no share exists.
        """
        control = self.rmse_control
        if control == 0.0:
            raise InvalidPairedUnitErrorsError("the control makes no error to reduce")
        return self.reduction / control

    def reduction_over(self, picks: Sequence[int]) -> float:
        """The reduction over the units at ``picks``, a unit counted as often as it is picked."""
        return self.rmse_control_over(picks) - self.rmse_candidate_over(picks)

    def rmse_control_over(self, picks: Sequence[int]) -> float:
        return _rmse_over(self.control, picks)

    def rmse_candidate_over(self, picks: Sequence[int]) -> float:
        return _rmse_over(self.candidate, picks)


def _rmse_over(errors: Sequence[UnitError], picks: Sequence[int]) -> float:
    squared = sum(errors[pick].squared_error for pick in picks)
    windows = sum(errors[pick].windows for pick in picks)
    return sqrt(squared / windows)


def _summed(repeats: Sequence[Sequence[UnitError]], side: str) -> tuple[UnitError, ...]:
    if not repeats:
        raise InvalidPairedUnitErrorsError(f"the {side} side has no repeat to pool")
    first = tuple(repeats[0])
    units = tuple(error.unit for error in first)
    squared = [error.squared_error for error in first]
    windows = [error.windows for error in first]
    for repeat in repeats[1:]:
        if tuple(error.unit for error in repeat) != units:
            raise InvalidPairedUnitErrorsError(
                f"the repeats of the {side} side disagree about the units"
            )
        for index, error in enumerate(repeat):
            squared[index] += error.squared_error
            windows[index] += error.windows
    return tuple(
        UnitError(unit=unit, squared_error=total, windows=count)
        for unit, total, count in zip(units, squared, windows, strict=True)
    )
