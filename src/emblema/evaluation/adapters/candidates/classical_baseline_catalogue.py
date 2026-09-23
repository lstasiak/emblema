from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.exceptions import UnknownCandidateError


class ClassicalBaselineCatalogue:
    """What the baselines are, without the means of fitting any of them.

    These candidates declare no compute budget. Holding a fit of trees to a transformer's epochs
    and batches is not a defined operation, so what each one spent is reported by the cell that
    ran it and the shared figure is left to the arms that can share one. How hard they fit is
    readable because whoever fits one fits it by the knobs that were declared.
    """

    def __init__(self, arms: Sequence[ClassicalArm], boosting: GradientBoostingSpec) -> None:
        self._arms = tuple(arms)
        self.boosting = boosting

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.CLASSICAL,
            starts_from=None,
            budget=None,
            method=self._method(self.arm_of(candidate)),
        )

    def arm_of(self, candidate: CandidateRef) -> ClassicalArm:
        """The baseline the campaign calls ``candidate``.

        Raises:
            UnknownCandidateError: If this catalogue holds no baseline of that name.
        """
        for arm in self._arms:
            if arm.ref == candidate:
                return arm
        raise UnknownCandidateError(f"this provider supplies no candidate {candidate}")

    def _method(self, arm: ClassicalArm) -> CandidateMethod:
        """What the baseline was set to, beyond the seed each cell of the grid supplies itself."""
        return CandidateMethod.of(
            features=arm.features,
            sources=" ".join(str(source) for source in arm.sources),
            rounds=self.boosting.rounds,
            max_depth=self.boosting.max_depth,
            learning_rate=self.boosting.learning_rate,
            row_share=self.boosting.row_share,
            feature_share=self.boosting.feature_share,
            min_leaf_weight=self.boosting.min_leaf_weight,
            l2_penalty=self.boosting.l2_penalty,
            threads=self.boosting.threads,
        )
