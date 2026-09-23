from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.task.run_purpose import RunPurpose


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
        declared: The candidate as the campaign recorded it — what it starts from, the
            arithmetic it was held to, and what it was set to. Carried whole rather than field
            by field, because a process configured otherwise would answer a point of the curve
            under something the grid never declared, and a check that named the fields would
            have to be remembered again for every field anyone adds.
    """

    task: TaskId
    cell: CampaignCell
    purpose: RunPurpose
    retain: bool
    declared: CampaignCandidate
