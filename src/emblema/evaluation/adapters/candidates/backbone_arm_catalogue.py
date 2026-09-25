from collections.abc import Sequence
from dataclasses import replace

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import (
    InvalidCandidateVariantError,
    UnknownCandidateError,
    UnknownKnobError,
)
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


class BackboneArmCatalogue:
    """What the arms of one backbone are, without the means of running any of them.

    Every arm is declared under its own schedule, and a variant of an arm is the arm with a
    knob of that schedule turned, read out of its name alone, so the process that declares a
    campaign and the one that runs its cells read the same variant out of the same text. The
    knobs leave the compute budget alone, which is how the campaign's promise that its neural
    candidates spend the same budget survives tuning: a variant has its base's budget by
    construction. What the schedule and the low-rank updates were set to travels with each
    candidate, so a campaign stored a month ago still says it.

    A backbone reaches a campaign as an artifact reference and a name, never as a checkpoint, a
    run or an experiment, so nothing here names the context the weights came from — and nothing
    here loads the stack that would turn them back into an encoder.
    """

    def __init__(self, arms: Sequence[BackboneArm]) -> None:
        self._arms = tuple(arms)

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        arm = self.arm_of(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=arm.backbone,
            budget=ComputeBudget.of(arm.schedule),
            method=self._method(arm),
        )

    def arm_of(self, candidate: CandidateRef) -> BackboneArm:
        """The arm the campaign calls ``candidate``, its schedule turned as the name says.

        Raises:
            UnknownCandidateError: If this catalogue holds no arm of that name, the name is not
                an arm and its knobs in name order, or a knob cannot be turned so.
        """
        try:
            variant = CandidateVariant.parse(candidate)
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        for arm in self._arms:
            if arm.ref == variant.base:
                return self._turned(arm, variant)
        raise UnknownCandidateError(f"this catalogue holds no arm called {variant.base}")

    @staticmethod
    def _turned(arm: BackboneArm, variant: CandidateVariant) -> BackboneArm:
        try:
            schedule = variant.applied_to(arm.schedule, AdaptationSchedule.tuned)
        except (UnknownKnobError, InvalidCandidateVariantError) as error:
            raise UnknownCandidateError(f"{variant.ref} names no variant: {error}") from error
        return replace(arm, ref=variant.ref, schedule=schedule)

    @staticmethod
    def _method(arm: BackboneArm) -> CandidateMethod:
        """What the arm was set to, beyond the arithmetic its budget allows.

        The counts the budget already carries are left out; what is here is everything a reader
        would otherwise have to take from whichever copy of the configuration still exists. The
        arm that starts from no weights names the model it is shaped like, since nothing else
        about it would say what size of network the control was.
        """
        stated: dict[str, object] = {
            "transfer_mode": arm.mode,
            "learning_rate": arm.schedule.learning_rate,
            "weight_decay": arm.schedule.weight_decay,
            "warmup_fraction": arm.schedule.warmup_fraction,
            "final_lr_fraction": arm.schedule.final_lr_fraction,
        }
        if arm.backbone is None:
            stated["architecture_of"] = f"{arm.architecture.key}@{arm.architecture.checksum}"
        if arm.lora is not None:
            stated["lora_rank"] = arm.lora.rank
            stated["lora_alpha"] = arm.lora.alpha
            stated["lora_dropout"] = arm.lora.dropout
            stated["lora_targets"] = ", ".join(arm.lora.targets)
        return CandidateMethod.of(**stated)
