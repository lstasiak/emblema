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


class ClassicalCandidateProvider:
    """Fits the baselines a catalogue names, in this process.

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
        """Fit one cell of a campaign in this process, as the variant the cell was declared to run.

        Raises:
            UnknownCandidateError: If the catalogue holds no baseline of that name.
            CandidateMismatchError: If the campaign recorded the baseline as anything other
                than what this process supplies — another method, other knobs or sources.
        """
        cell = request.cell
        runs = request.declared.ref
        request.declared.must_match(self._catalogue.describe(runs))
        arm = self._catalogue.arm_of(runs)
        outcome = self._run(
            RunClassicalFitCommand(
                task=request.task,
                recipe=ClassicalRecipe(method=arm.method, seed=cell.seed, sources=arm.sources),
                budget=cell.budget,
                sample_seed=cell.seed,
                purpose=request.purpose,
                retain=request.retain,
                holdout=request.holdout,
            )
        )
        return CellResult.of(cell, outcome)
