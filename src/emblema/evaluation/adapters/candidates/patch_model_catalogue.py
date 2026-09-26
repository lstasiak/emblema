from dataclasses import dataclass, replace
from typing import Self

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
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


@dataclass(frozen=True, kw_only=True)
class _PatchModel:
    """What the patch model is set to: its shape, its schedule and its pooling, each turnable."""

    spec: PatchModelSpec
    schedule: AdaptationSchedule
    pooling: HeadPooling

    def tuned(self, knob: str, value: str) -> Self:
        """This setting with ``knob`` turned, on whichever of the three has it.

        Raises:
            UnknownKnobError: If none of them has such a knob, or it cannot take that value.
        """
        if knob in HeadPooling.KNOBS:
            return replace(self, pooling=self.pooling.tuned(knob, value))
        if knob in PatchModelSpec.KNOBS:
            return replace(self, spec=self.spec.tuned(knob, value))
        return replace(self, schedule=self.schedule.tuned(knob, value))


class PatchModelCatalogue:
    """What the patch model is, without the means of training it.

    A network, so it is held to the compute budget the adapted arms share, and it learns under
    their schedule: one schedule declared, one run under, and the budgets equal because they are
    derived from it the same way. A variant of the model is the model under a turned schedule,
    the same knobs the arms turn, under a turned pooling, or in a turned shape — none of which
    touches the budget, which is the schedule's epochs, floor of steps and batch alone. The
    shape, the schedule and the pooling it was set to travel with the candidate, so a campaign
    stored a month ago still says what it compared.
    """

    def __init__(
        self,
        ref: CandidateRef,
        spec: PatchModelSpec,
        schedule: AdaptationSchedule,
        pooling: HeadPooling | None = None,
    ) -> None:
        self._ref = ref
        self._model = _PatchModel(
            spec=spec,
            schedule=schedule,
            pooling=HeadPooling.mean() if pooling is None else pooling,
        )

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        model = self._model_of(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=None,
            budget=ComputeBudget.of(model.schedule),
            method=CandidateMethod.of(
                **model.spec.parameters(),
                learning_rate=model.schedule.learning_rate,
                weight_decay=model.schedule.weight_decay,
                warmup_fraction=model.schedule.warmup_fraction,
                final_lr_fraction=model.schedule.final_lr_fraction,
                **model.pooling.parameters(),
            ),
        )

    def plan_of(self, candidate: CandidateRef, seed: int) -> PatchPlan:
        """How the model the campaign calls ``candidate`` is trained under ``seed``.

        Raises:
            UnknownCandidateError: If this catalogue holds no model of that name.
        """
        model = self._model_of(candidate)
        return PatchPlan(spec=model.spec, schedule=model.schedule, seed=seed, pooling=model.pooling)

    def schedule_of(self, candidate: CandidateRef) -> AdaptationSchedule:
        """The schedule the model the campaign calls ``candidate`` learns under.

        Raises:
            UnknownCandidateError: If this catalogue holds no model of that name, the name is not
                the model and its knobs in name order, or a knob cannot be turned so.
        """
        return self._model_of(candidate).schedule

    def _model_of(self, candidate: CandidateRef) -> _PatchModel:
        try:
            variant = CandidateVariant.parse(candidate)
        except InvalidCandidateVariantError as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
        if variant.base != self._ref:
            raise UnknownCandidateError(
                f"this catalogue holds the patch model {self._ref}, not {variant.base}"
            )
        try:
            return variant.applied_to(self._model, _PatchModel.tuned)
        except (UnknownKnobError, InvalidCandidateVariantError) as error:
            raise UnknownCandidateError(f"{candidate} names no variant: {error}") from error
