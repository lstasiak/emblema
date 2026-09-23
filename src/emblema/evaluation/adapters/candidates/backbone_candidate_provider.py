from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import UnknownBackboneError, UnknownCandidateError
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


class BackboneCandidateProvider:
    """Supplies a campaign with the arms that adapt one pretrained backbone.

    A backbone reaches a campaign as an artifact reference and a name, never as a checkpoint, a
    run or an experiment: what this needs of pretrained weights is a reference and a checksum,
    which is exactly what the published language carries. Turning those bytes back into an
    encoder is the process's business, behind the runtime this is given, so nothing here names
    the context the weights came from.

    Every arm shares one schedule, which is how the campaign's promise that its neural
    candidates spend the same compute budget is kept rather than merely stated: there is one
    schedule to declare and one to run under. What the schedule and the low-rank updates were
    set to travels with each candidate, so a campaign stored a month ago still says it.
    """

    def __init__(
        self,
        arms: Sequence[BackboneArm],
        schedule: AdaptationSchedule,
        run_adaptation: RunAdaptation,
    ) -> None:
        self._arms = tuple(arms)
        self._schedule = schedule
        self._run = run_adaptation

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        arm = self._arm(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=arm.backbone,
            budget=ComputeBudget(
                epochs=self._schedule.epochs,
                min_steps=self._schedule.min_steps,
                batch_size=self._schedule.batch_size,
            ),
            method=self._method(arm),
        )

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Run one cell of a campaign on this machine.

        Raises:
            UnknownCandidateError: If this provider supplies no arm of that name.
            UnknownBackboneError: If the campaign recorded the arm as starting from other
                weights than this provider was built over.
        """
        cell = request.cell
        arm = self._arm(cell.candidate)
        if request.starts_from != arm.backbone:
            raise UnknownBackboneError(
                f"the campaign ran {cell.candidate} over other weights than this provider serves"
            )
        outcome = self._run(
            RunAdaptationCommand(
                task=request.task,
                plan=AdaptationPlan(
                    mode=arm.mode,
                    backbone=arm.backbone,
                    schedule=self._schedule,
                    lora=arm.lora,
                    seed=cell.seed,
                ),
                budget=cell.budget,
                sample_seed=cell.seed,
                purpose=request.purpose,
                retain=request.retain,
            )
        )
        return CellResult(
            cell=cell,
            errors=outcome.by_unit(),
            seconds=outcome.seconds,
            artifact=outcome.artifact,
        )

    def _method(self, arm: BackboneArm) -> CandidateMethod:
        """What the arm was set to, beyond the arithmetic its budget allows.

        The counts the budget already carries are left out; what is here is everything a reader
        would otherwise have to take from whichever copy of the configuration still exists.
        """
        stated: dict[str, object] = {
            "transfer_mode": arm.mode,
            "learning_rate": self._schedule.learning_rate,
            "weight_decay": self._schedule.weight_decay,
            "warmup_fraction": self._schedule.warmup_fraction,
            "final_lr_fraction": self._schedule.final_lr_fraction,
        }
        if arm.lora is not None:
            stated["lora_rank"] = arm.lora.rank
            stated["lora_alpha"] = arm.lora.alpha
            stated["lora_dropout"] = arm.lora.dropout
            stated["lora_targets"] = ", ".join(arm.lora.targets)
        return CandidateMethod.of(**stated)

    def _arm(self, candidate: CandidateRef) -> BackboneArm:
        """The arm the campaign calls ``candidate``.

        Raises:
            UnknownCandidateError: If this provider supplies no arm of that name.
        """
        for arm in self._arms:
            if arm.ref == candidate:
                return arm
        raise UnknownCandidateError(f"this provider supplies no candidate {candidate}")
