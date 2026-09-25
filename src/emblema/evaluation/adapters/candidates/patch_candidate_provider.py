from emblema.evaluation.adapters.candidates.patch_model_catalogue import PatchModelCatalogue
from emblema.evaluation.application.use_cases.run_patch_training import (
    RunPatchTraining,
    RunPatchTrainingCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult


class PatchCandidateProvider:
    """Trains the patch model a catalogue names, on this machine.

    What the model is comes from the catalogue and not from here, so the description a campaign
    was designed against and the one a cell is checked against are the same text.
    """

    def __init__(
        self, catalogue: PatchModelCatalogue, run_patch_training: RunPatchTraining
    ) -> None:
        self._catalogue = catalogue
        self._run = run_patch_training

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._catalogue.describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Run one cell of a campaign on this machine.

        Raises:
            UnknownCandidateError: If the catalogue holds no model of that name.
            CandidateMismatchError: If the campaign recorded the model as anything other than
                what this process supplies — another shape, budget or schedule.
        """
        cell = request.cell
        request.declared.must_match(self._catalogue.describe(cell.candidate))
        outcome = self._run(
            RunPatchTrainingCommand(
                task=request.task,
                plan=self._catalogue.plan_of(cell.candidate, cell.seed),
                budget=cell.budget,
                sample_seed=cell.seed,
                purpose=request.purpose,
                retain=request.retain,
                holdout=request.holdout,
            )
        )
        return CellResult.of(cell, outcome)
