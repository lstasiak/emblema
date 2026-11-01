from collections.abc import Sequence
from dataclasses import replace

from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    InvalidCandidateVariantError,
    InvalidDetectorRecipeError,
    UnknownCandidateError,
    UnknownKnobError,
)
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


class DetectorCatalogue:
    """What the detectors are, without the means of fitting any of them.

    A detector declares no compute budget: it takes no step, and what it spent is reported by
    the cell that ran it. A variant of a detector is read out of its name alone, so the process
    that declares a campaign and the one that runs its cells read the same variant out of the
    same text.
    """

    def __init__(self, arms: Sequence[DetectorArm]) -> None:
        self._arms = tuple(arms)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        arm = self.arm_of(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.DETECTOR,
            starts_from=arm.backbone,
            budget=None,
            method=self._method(arm),
        )

    def arm_of(self, candidate: CandidateRef) -> DetectorArm:
        """The detector the campaign calls ``candidate``, its knobs turned as the name says.

        Raises:
            UnknownCandidateError: If this catalogue holds no detector of that name, the name
                is not a detector and its knobs in name order, or a knob cannot be turned so.
        """
        try:
            variant = CandidateVariant.parse(candidate)
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        for arm in self._arms:
            if arm.ref == variant.base:
                return self._turned(arm, variant)
        raise UnknownCandidateError(f"this catalogue holds no detector called {variant.base}")

    @staticmethod
    def _turned(arm: DetectorArm, variant: CandidateVariant) -> DetectorArm:
        try:
            turned = variant.applied_to(arm, DetectorArm.tuned)
        except (
            UnknownKnobError,
            InvalidCandidateVariantError,
            InvalidDetectorRecipeError,
        ) as error:
            raise UnknownCandidateError(f"{variant.ref} names no variant: {error}") from error
        return replace(turned, ref=variant.ref)

    @staticmethod
    def _method(arm: DetectorArm) -> CandidateMethod:
        """What the detector was set to, beyond the seed each cell of the grid supplies itself.

        An arm over the summaries or their counts names what it reads and the detector's knobs,
        the reference and the distance only where they are not the defaults, so the campaigns
        declared before they were knobs read as they were stored; one over an encoder's states
        names the model it is shaped like too, the layer and the pooling it reads at; one that
        asks the pretext names the draws in place of the neighbours, the backbone whose masks it
        asks under, which cells it hides where not the backbone's own draw and how it reads the
        errors where not raw; one by a distance other than the nearest neighbours' names the
        distance and no neighbours.
        """
        stated: dict[str, object] = {"features": arm.features}
        if arm.features.asks_the_pretext and arm.backbone is not None:
            stated["draws"] = arm.draws
            stated["masks"] = arm.backbone.key
            if arm.hidden is not HiddenCellsChoice.OWN:
                stated["hidden"] = arm.hidden
            if arm.normalisation is not ErrorNormalisation.RAW:
                stated["error"] = arm.normalisation
        elif arm.distance is not ScoreDistance.NEAREST_NEIGHBOURS:
            stated["distance"] = arm.distance
        else:
            stated["neighbours"] = arm.neighbours
        stated["contamination"] = arm.contamination
        if arm.reference is not ScoreReference.FITTING_SIDE:
            stated["reference"] = arm.reference
        if arm.architecture is not None:
            stated["architecture"] = arm.architecture.key
            stated["layer"] = arm.encoder.layer
            stated.update(arm.pooling.parameters())
        return CandidateMethod.of(**stated)
