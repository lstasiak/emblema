from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


class PatchModelCatalogue:
    """What the patch model is, without the means of training it.

    A network, so it is held to the compute budget the adapted arms share, and it learns under
    their schedule: one schedule declared, one run under, and the budgets equal because they are
    derived from it the same way. The shape and the schedule it was set to travel with the
    candidate, so a campaign stored a month ago still says what it compared.

    It has no variants yet: a name that turns a knob is refused rather than read, because how
    the knobs of a network are tuned is decided for every network at once and not here.
    """

    def __init__(
        self, ref: CandidateRef, spec: PatchModelSpec, schedule: AdaptationSchedule
    ) -> None:
        self._ref = ref
        self._spec = spec
        self._schedule = schedule

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        self._must_hold(candidate)
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.NEURAL,
            starts_from=None,
            budget=ComputeBudget.of(self._schedule),
            method=CandidateMethod.of(
                **self._spec.parameters(),
                learning_rate=self._schedule.learning_rate,
                weight_decay=self._schedule.weight_decay,
                warmup_fraction=self._schedule.warmup_fraction,
                final_lr_fraction=self._schedule.final_lr_fraction,
            ),
        )

    def plan_of(self, candidate: CandidateRef, seed: int) -> PatchPlan:
        """How the model the campaign calls ``candidate`` is trained under ``seed``.

        Raises:
            UnknownCandidateError: If this catalogue holds no model of that name.
        """
        self._must_hold(candidate)
        return PatchPlan(spec=self._spec, schedule=self._schedule, seed=seed)

    def _must_hold(self, candidate: CandidateRef) -> None:
        if candidate != self._ref:
            raise UnknownCandidateError(
                f"this catalogue holds the patch model {self._ref} and no variant of it, "
                f"not {candidate}"
            )
