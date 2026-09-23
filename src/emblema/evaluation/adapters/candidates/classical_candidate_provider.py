from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.application.use_cases.run_classical_fit import (
    RunClassicalFit,
    RunClassicalFitCommand,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.exceptions import UnknownBackboneError, UnknownCandidateError


class ClassicalCandidateProvider:
    """Supplies a campaign with the baselines that learn a task from its labels alone.

    Nothing here reaches the context that trains backbones, and that is the point rather than an
    accident of what this happens to need: a campaign made only of these runs end to end with
    the Pretraining context absent, which is what makes the comparison between the two kinds a
    comparison and not a report about one of them.

    These candidates declare no compute budget. Holding a fit of trees to a transformer's
    epochs and batches is not a defined operation, so what each one spent is reported by the
    cell that ran it and the shared figure is left to the arms that can share one.
    """

    def __init__(
        self,
        arms: Sequence[ClassicalArm],
        boosting: GradientBoostingSpec,
        run_classical_fit: RunClassicalFit,
    ) -> None:
        self._arms = tuple(arms)
        self._boosting = boosting
        self._run = run_classical_fit

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return CampaignCandidate(
            ref=candidate,
            kind=CandidateKind.CLASSICAL,
            starts_from=None,
            budget=None,
            method=self._method(self._arm(candidate)),
        )

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Fit one cell of a campaign in this process.

        Raises:
            UnknownCandidateError: If this provider supplies no baseline of that name.
            UnknownBackboneError: If the campaign recorded the baseline as starting from
                pretrained weights, which nothing here could have used.
        """
        cell = request.cell
        arm = self._arm(cell.candidate)
        if request.starts_from is not None:
            raise UnknownBackboneError(
                f"the campaign recorded {cell.candidate} as starting from weights, and a "
                f"baseline starts from none"
            )
        outcome = self._run(
            RunClassicalFitCommand(
                task=request.task,
                recipe=ClassicalRecipe(
                    features=arm.features,
                    boosting=self._boosting,
                    seed=cell.seed,
                    sources=arm.sources,
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

    def _method(self, arm: ClassicalArm) -> CandidateMethod:
        """What the baseline was set to, beyond the seed each cell of the grid supplies itself."""
        return CandidateMethod.of(
            features=arm.features,
            sources=" ".join(str(source) for source in arm.sources),
            rounds=self._boosting.rounds,
            max_depth=self._boosting.max_depth,
            learning_rate=self._boosting.learning_rate,
            row_share=self._boosting.row_share,
            feature_share=self._boosting.feature_share,
            min_leaf_weight=self._boosting.min_leaf_weight,
            l2_penalty=self._boosting.l2_penalty,
            threads=self._boosting.threads,
        )

    def _arm(self, candidate: CandidateRef) -> ClassicalArm:
        """The baseline the campaign calls ``candidate``.

        Raises:
            UnknownCandidateError: If this provider supplies no baseline of that name.
        """
        for arm in self._arms:
            if arm.ref == candidate:
                return arm
        raise UnknownCandidateError(f"this provider supplies no candidate {candidate}")
