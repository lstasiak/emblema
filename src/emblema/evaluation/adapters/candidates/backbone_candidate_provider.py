from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import (
    CandidateMethodMismatchError,
    UnknownBackboneError,
)
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan


class BackboneCandidateProvider:
    """Runs the arms a catalogue names, on this machine.

    What each arm is comes from the catalogue and not from here, so the description a campaign
    was designed against and the one a cell is checked against are the same text. Turning
    pretrained weights back into an encoder is the process's business, behind the runtime this
    is given, so nothing here names the context the weights came from.
    """

    def __init__(self, catalogue: BackboneArmCatalogue, run_adaptation: RunAdaptation) -> None:
        self._catalogue = catalogue
        self._run = run_adaptation

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        return self._catalogue.describe(candidate)

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Run one cell of a campaign on this machine.

        Raises:
            UnknownCandidateError: If the catalogue holds no arm of that name.
            UnknownBackboneError: If the campaign recorded the arm as starting from other
                weights than this provider was built over.
            CandidateMethodMismatchError: If it recorded the arm under another schedule or
                another low-rank update than this process runs.
        """
        cell = request.cell
        declared = self._catalogue.describe(cell.candidate)
        arm = self._catalogue.arm_of(cell.candidate)
        if request.starts_from != arm.backbone:
            raise UnknownBackboneError(
                f"the campaign ran {cell.candidate} over other weights than this provider serves"
            )
        if request.method != declared.method:
            raise CandidateMethodMismatchError(
                f"the campaign recorded {cell.candidate} under other settings than this process "
                f"holds: {request.method.parameters} against {declared.method.parameters}"
            )
        outcome = self._run(
            RunAdaptationCommand(
                task=request.task,
                plan=AdaptationPlan(
                    mode=arm.mode,
                    backbone=arm.backbone,
                    schedule=self._catalogue.schedule,
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
