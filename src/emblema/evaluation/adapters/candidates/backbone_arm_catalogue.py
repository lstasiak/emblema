from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


class BackboneArmCatalogue:
    """What the arms of one backbone are, without the means of running any of them.

    Every arm shares one schedule, which is how the campaign's promise that its neural
    candidates spend the same compute budget is kept rather than merely stated: there is one
    schedule to declare and one to run under. What the schedule and the low-rank updates were
    set to travels with each candidate, so a campaign stored a month ago still says it. The
    schedule is readable because whoever runs an arm runs it under the one that was declared.

    A backbone reaches a campaign as an artifact reference and a name, never as a checkpoint, a
    run or an experiment, so nothing here names the context the weights came from — and nothing
    here loads the stack that would turn them back into an encoder.
    """

    def __init__(self, arms: Sequence[BackboneArm], schedule: AdaptationSchedule) -> None:
        self._arms = tuple(arms)
        self.schedule = schedule

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        arm = self.arm_of(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=arm.backbone,
            budget=ComputeBudget.of(self.schedule),
            method=self._method(arm),
        )

    def arm_of(self, candidate: CandidateRef) -> BackboneArm:
        """The arm the campaign calls ``candidate``.

        Raises:
            UnknownCandidateError: If this catalogue holds no arm of that name.
        """
        for arm in self._arms:
            if arm.ref == candidate:
                return arm
        raise UnknownCandidateError(f"this catalogue holds no arm called {candidate}")

    def _method(self, arm: BackboneArm) -> CandidateMethod:
        """What the arm was set to, beyond the arithmetic its budget allows.

        The counts the budget already carries are left out; what is here is everything a reader
        would otherwise have to take from whichever copy of the configuration still exists.
        """
        stated: dict[str, object] = {
            "transfer_mode": arm.mode,
            "learning_rate": self.schedule.learning_rate,
            "weight_decay": self.schedule.weight_decay,
            "warmup_fraction": self.schedule.warmup_fraction,
            "final_lr_fraction": self.schedule.final_lr_fraction,
        }
        if arm.lora is not None:
            stated["lora_rank"] = arm.lora.rank
            stated["lora_alpha"] = arm.lora.alpha
            stated["lora_dropout"] = arm.lora.dropout
            stated["lora_targets"] = ", ".join(arm.lora.targets)
        return CandidateMethod.of(**stated)
