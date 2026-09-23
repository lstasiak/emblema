from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.application.use_cases.run_classical_fit import (
    RunClassicalFit,
    RunClassicalFitCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.exceptions import (
    CandidateMethodMismatchError,
    UnknownBackboneError,
)


class ClassicalCandidateProvider:
    """Fits the baselines a catalogue names, in this process.

    Nothing here reaches the context that trains backbones, and that is the point rather than an
    accident of what it happens to need: a campaign made only of these runs end to end with the
    Pretraining context absent, which is what makes the comparison between the two kinds a
    comparison and not a report about one of them.

    What each baseline is comes from the catalogue and not from here, so the description a
    campaign was designed against and the one a cell is checked against are the same text.
    """

    def __init__(
        self, catalogue: ClassicalBaselineCatalogue, run_classical_fit: RunClassicalFit
    ) -> None:
        self._catalogue = catalogue
        self._run = run_classical_fit

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._catalogue.describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Fit one cell of a campaign in this process.

        Raises:
            UnknownCandidateError: If the catalogue holds no baseline of that name.
            UnknownBackboneError: If the campaign recorded the baseline as starting from
                pretrained weights, which nothing here could have used.
            CandidateMethodMismatchError: If it recorded the baseline under other boosting knobs
                than this process fits by.
        """
        cell = request.cell
        declared = self._catalogue.describe(cell.candidate)
        arm = self._catalogue.arm_of(cell.candidate)
        if request.starts_from is not None:
            raise UnknownBackboneError(
                f"the campaign recorded {cell.candidate} as starting from weights, and a "
                f"baseline starts from none"
            )
        if request.method != declared.method:
            raise CandidateMethodMismatchError(
                f"the campaign recorded {cell.candidate} under other settings than this process "
                f"holds: {request.method.parameters} against {declared.method.parameters}"
            )
        outcome = self._run(
            RunClassicalFitCommand(
                task=request.task,
                recipe=ClassicalRecipe(
                    features=arm.features,
                    boosting=self._catalogue.boosting,
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
