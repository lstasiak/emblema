from collections.abc import Sequence
from dataclasses import replace

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.exceptions import (
    InvalidCandidateVariantError,
    UnknownCandidateError,
    UnknownKnobError,
)
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


class ClassicalBaselineCatalogue:
    """What the baselines are, without the means of fitting any of them.

    These candidates declare no compute budget. Holding a fit of trees to a transformer's epochs
    and batches is not a defined operation, so what each one spent is reported by the cell that
    ran it and the shared figure is left to the arms that can share one. How hard they fit is
    readable because whoever fits one fits it by the knobs that were declared.
    """

    def __init__(self, arms: Sequence[ClassicalArm]) -> None:
        self._arms = tuple(arms)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.CLASSICAL,
            starts_from=None,
            budget=None,
            method=self._method(self.arm_of(candidate)),
        )

    def arm_of(self, candidate: CandidateRef) -> ClassicalArm:
        """The baseline the campaign calls ``candidate``, its knobs turned as the name says.

        A variant is read out of its name alone, so every process holding this catalogue reads
        the same variant out of the same text.

        Raises:
            UnknownCandidateError: If this catalogue holds no baseline of that name, the name is
                not a baseline and its knobs in name order, or a knob cannot be turned so.
        """
        try:
            variant = CandidateVariant.parse(candidate)
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        for arm in self._arms:
            if arm.ref == variant.base:
                return self._turned(arm, variant)
        raise UnknownCandidateError(f"this catalogue holds no baseline called {variant.base}")

    @staticmethod
    def _turned(arm: ClassicalArm, variant: CandidateVariant) -> ClassicalArm:
        method = arm.method
        for knob, value in variant.knobs:
            try:
                method = method.tuned(knob, value)
            except UnknownKnobError as error:
                raise UnknownCandidateError(f"{variant.ref} names no variant: {error}") from error
        if variant.knobs and method == arm.method:
            # Two names for one model would let a selection weigh the default against itself.
            raise UnknownCandidateError(f"{variant.ref} turns no knob away from {arm.ref}")
        return replace(arm, ref=variant.ref, method=method)

    def _method(self, arm: ClassicalArm) -> CandidateMethod:
        """What the baseline was set to, beyond the seed each cell of the grid supplies itself."""
        return CandidateMethod.of(
            sources=" ".join(str(source) for source in arm.sources), **arm.method.parameters()
        )
