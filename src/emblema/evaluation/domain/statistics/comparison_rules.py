from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidComparisonRulesError
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.family_correction import FamilyCorrection
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind


@dataclass(frozen=True, kw_only=True)
class ComparisonRules:
    """The registered rules every cell of a curve is judged by, stated once before the grid.

    The endpoint stands alone: it is confirmed when the reduction is at least the registered
    minimum, its whole interval lies above zero and it clears the practical
    floor — a reduction the floor swallows is practically nil whatever its interval says, so a
    confirmation under the floor would contradict the registration's own words. A secondary
    cell is judged by the family's correction and not by its own interval, so that its verdict
    and its rejection never disagree: rejected, positive and above the floor it is
    distinguishable; rejected but under the floor practically nil; rejected the wrong way
    worse; not rejected indistinguishable. The family is the registered one whatever has run:
    a cell that has not run enters with a p-value of one, which is never rejected and holds the
    others to the levels the registration set, so a partial grid is read more strictly than the
    whole one and never less. Which correction the family is read under is part of the rules,
    named before the grid runs like everything else here: a family read under Holm and again
    under a looser correction once the numbers are in would be choosing its own answer.

    Both thresholds — the endpoint's minimum and the floor's fixed part — are stated the same
    way, as a share of the control's error or as an amount in its unit, so the registration is
    read in one currency.

    Invariants: a relative minimum lies in ``(0, 1)`` and an absolute one is positive and
    finite; the floor's fixed part is finite and not negative; the secondary family has at
    least one member.

    Attributes:
        minimum_reduction: What the endpoint's reduction must reach.
        floor_part: The fixed part of the practical floor.
        threshold: Whether both are shares of the control's error or amounts in its unit.
        correction: The correction the secondary family is tested under.
        secondary_family_size: How many secondary cells the registration names.
    """

    minimum_reduction: float
    floor_part: float
    threshold: ThresholdKind
    correction: FamilyCorrection
    secondary_family_size: int

    def __post_init__(self) -> None:
        match self.threshold:
            case ThresholdKind.RELATIVE:
                valid = 0.0 < self.minimum_reduction < 1.0
            case ThresholdKind.ABSOLUTE:
                valid = isfinite(self.minimum_reduction) and self.minimum_reduction > 0.0
        if not valid:
            raise InvalidComparisonRulesError(
                f"a {self.threshold} minimum_reduction of {self.minimum_reduction} is not a "
                "reduction anything could reach"
            )
        if not isfinite(self.floor_part) or self.floor_part < 0.0:
            raise InvalidComparisonRulesError(
                f"floor_part must be finite and not negative, got {self.floor_part}"
            )
        if self.secondary_family_size < 1:
            raise InvalidComparisonRulesError(
                f"the secondary family needs a member, got {self.secondary_family_size}"
            )

    def floor_of(self, control: ErrorOverRepeats) -> PracticalFloor:
        """The practical floor at one budget, from the control's error and its repeats."""
        return PracticalFloor.of(control, part=self.floor_part, threshold=self.threshold)

    def endpoint_verdict(
        self, difference: PairedDifference, floor: PracticalFloor
    ) -> ComparisonVerdict:
        """What the rules say about the endpoint, judged on its own interval."""
        if difference.confirms(self.minimum_reduction, self.threshold) and not floor.swallows(
            difference.reduction
        ):
            return ComparisonVerdict.CONFIRMED
        if not difference.distinguishable:
            return ComparisonVerdict.INDISTINGUISHABLE
        return self._against_the_floor(
            difference, floor, cleared=ComparisonVerdict.BELOW_REGISTERED_REDUCTION
        )

    def secondary_verdict(
        self, difference: PairedDifference, floor: PracticalFloor, *, rejected: bool
    ) -> ComparisonVerdict:
        """What the rules say about a secondary cell, ``rejected`` being the family's word."""
        if not rejected:
            return ComparisonVerdict.INDISTINGUISHABLE
        return self._against_the_floor(difference, floor, cleared=ComparisonVerdict.DISTINGUISHABLE)

    def secondary_rejections(self, p_values: Sequence[float]) -> tuple[bool, ...]:
        """Which of the secondary cells that ran are rejected, over the registered family.

        Raises:
            InvalidFamilyCorrectionError: If more cells ran than the family names, or a p-value
                lies outside ``[0, 1]``.
        """
        return self.correction.rejected(p_values, family_size=self.secondary_family_size)

    @staticmethod
    def _against_the_floor(
        difference: PairedDifference, floor: PracticalFloor, *, cleared: ComparisonVerdict
    ) -> ComparisonVerdict:
        """A distinguishable difference held to the floor: ``cleared`` if it clears it."""
        if difference.reduction < 0.0:
            return ComparisonVerdict.WORSE
        if floor.swallows(difference.reduction):
            return ComparisonVerdict.PRACTICALLY_NIL
        return cleared
