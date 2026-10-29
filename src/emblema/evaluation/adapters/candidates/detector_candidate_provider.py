from emblema.evaluation.adapters.candidates.detector_catalogue import DetectorCatalogue
from emblema.evaluation.application.use_cases.run_detection import (
    RunDetection,
    RunDetectionCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import ProtocolMismatchError


class DetectorCandidateProvider:
    """Fits the detectors a catalogue names, in this process.

    What each detector is comes from the catalogue and not from here, so the description a
    campaign was designed against and the one a cell is checked against are the same text. A
    detector's cell has one budget, every window of the fitting side, and no division of the
    tuning side: a selection among detectors would choose on labels the task says it has not
    got, so a selection run is refused before anything is read.
    """

    def __init__(self, catalogue: DetectorCatalogue, run_detection: RunDetection) -> None:
        self._catalogue = catalogue
        self._run = run_detection

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._catalogue.describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Fit one cell of a campaign in this process, as the variant the cell was declared to run.

        Raises:
            UnknownCandidateError: If the catalogue holds no detector of that name.
            CandidateMismatchError: If the campaign recorded the detector as anything other
                than what this process supplies.
            ProtocolMismatchError: If the cell is a selection run, or its budget is not the
                one of every window.
        """
        cell = request.cell
        runs = request.declared.ref
        request.declared.must_match(self._catalogue.describe(runs))
        if request.holdout is not None:
            raise ProtocolMismatchError(
                "a detector's settings are declared before the run; nothing is selected on the "
                "tuning side of a task that spends no labels"
            )
        if cell.budget.windows is not None:
            raise ProtocolMismatchError(
                f"a detector fits on every window of the fitting side, not on {cell.budget.text()}"
            )
        arm = self._catalogue.arm_of(runs)
        outcome = self._run(
            RunDetectionCommand(
                task=request.task,
                recipe=arm.recipe(seed=cell.seed),
                purpose=request.purpose,
                retain=request.retain,
            )
        )
        return CellResult.of(cell, outcome)
