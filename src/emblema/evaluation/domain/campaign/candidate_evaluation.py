from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CandidateEvaluation:
    """A request to run one cell: which candidate, on what, under which rules of access.

    Everything a candidate needs to produce one point, and nothing about the campaign it belongs
    to: what has run already and what the grid holds are the campaign's business, and a
    candidate that could see them could behave differently late in a grid than early in it.

    Attributes:
        task: Task to learn and be scored on.
        cell: Which candidate, at which budget, under which seed.
        purpose: What the run is for, which decides whether it is scored on the validation side
            or on the frozen one.
        retain: Whether the fitted candidate is to be kept as an artifact. True for the one cell
            per candidate the campaign designated before anything ran.
        starts_from: The weights the campaign recorded this candidate as starting from, so a
            provider serving other ones refuses the cell instead of quietly answering it with
            the wrong backbone; ``None`` for a candidate that starts from none.
        method: What the campaign recorded this candidate as being set to. A process configured
            otherwise would answer a point of the curve under settings the grid never declared,
            and the grid would read as though one candidate had been compared at one setting.
    """

    task: TaskId
    cell: CampaignCell
    purpose: RunPurpose
    retain: bool
    starts_from: ArtifactRef | None
    method: CandidateMethod
