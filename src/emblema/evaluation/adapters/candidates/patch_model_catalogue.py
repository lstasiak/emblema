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
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


class PatchModelCatalogue:
    """What the patch model is, without the means of training it.

    A network, so it is held to the compute budget the adapted arms share, and it learns under
    their schedule: one schedule declared, one run under, and the budgets equal because they are
    derived from it the same way. A variant of the model is the model under a turned schedule,
    the same knobs the arms turn, so every network of a campaign is tuned by one protocol; the
    knobs of its shape are not turned here (ADR-0039). The shape and the schedule it was set to
    travel with the candidate, so a campaign stored a month ago still says what it compared.
    """

    def __init__(
        self, ref: CandidateRef, spec: PatchModelSpec, schedule: AdaptationSchedule
    ) -> None:
        self._ref = ref
        self._spec = spec
        self._schedule = schedule

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        schedule = self.schedule_of(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=None,
            budget=ComputeBudget.of(schedule),
            method=CandidateMethod.of(
                **self._spec.parameters(),
                learning_rate=schedule.learning_rate,
                weight_decay=schedule.weight_decay,
                warmup_fraction=schedule.warmup_fraction,
                final_lr_fraction=schedule.final_lr_fraction,
            ),
        )

    def plan_of(self, candidate: CandidateRef, seed: int) -> PatchPlan:
        """How the model the campaign calls ``candidate`` is trained under ``seed``.

        Raises:
            UnknownCandidateError: If this catalogue holds no model of that name.
        """
        return PatchPlan(spec=self._spec, schedule=self.schedule_of(candidate), seed=seed)

    def schedule_of(self, candidate: CandidateRef) -> AdaptationSchedule:
        """The schedule the model the campaign calls ``candidate`` learns under.

        Raises:
            UnknownCandidateError: If this catalogue holds no model of that name, the name is not
                the model and its knobs in name order, or a knob cannot be turned so.
        """
        try:
            variant = CandidateVariant.parse(candidate)
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        if variant.base != self._ref:
            raise UnknownCandidateError(
                f"this catalogue holds the patch model {self._ref}, not {variant.base}"
            )
        try:
            return variant.applied_to(self._schedule, AdaptationSchedule.tuned)
        except (UnknownKnobError, InvalidCandidateVariantError) as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
