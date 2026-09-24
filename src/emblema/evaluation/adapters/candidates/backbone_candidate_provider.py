from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan


class BackboneCandidateProvider:
    """Runs the arms a catalogue names, on this machine.

    What each arm is comes from the catalogue and not from here, so the description a campaign
    was designed against and the one a cell is checked against are the same text. Turning
    pretrained weights back into an encoder is the runtime's business, so nothing here names
    the context the weights came from.
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
            CandidateMismatchError: If the campaign recorded the arm as anything other than what
                this process supplies — other weights, another budget, another schedule or
                another low-rank update.
        """
        cell = request.cell
        request.declared.must_match(self._catalogue.describe(cell.candidate))
        arm = self._catalogue.arm_of(cell.candidate)
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
                holdout=request.holdout,
            )
        )
        return CellResult.of(cell, outcome)
