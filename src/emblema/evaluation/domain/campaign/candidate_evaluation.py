from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.exceptions import InvalidInnerHoldoutError
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
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
        declared: The candidate as the campaign recorded it for this cell — the variant a
            tuned candidate runs at this budget — carried whole rather than field by field, so
            a process configured otherwise is refused the cell rather than answering a point of
            the curve under something the grid never declared.
        holdout: How the tuning side is divided for a selection run; ``None`` for every other.

    Invariants: a selection run divides the tuning side, and no other run does.
    """

    task: TaskId
    cell: CampaignCell
    purpose: RunPurpose
    retain: bool
    declared: CampaignCandidate
    holdout: InnerHoldout | None = None

    def __post_init__(self) -> None:
        if self.purpose is RunPurpose.SELECTION and self.holdout is None:
            raise InvalidInnerHoldoutError(
                "a selection run divides the tuning side and was given no division of it"
            )
        if self.purpose is not RunPurpose.SELECTION and self.holdout is not None:
            raise InvalidInnerHoldoutError(
                f"a {self.purpose.value} run is scored outside the tuning side and was given a "
                "division of it"
            )
