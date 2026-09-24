from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.exceptions import (
    CampaignOrderRejectedError,
    InvalidCampaignOrderError,
)
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.task.downstream_task import DownstreamTask


@dataclass(frozen=True, kw_only=True)
class CampaignOrder:
    """Cells of a campaign handed to a machine that cannot reach the registry the campaign is in.

    Complete enough to run without asking anything back: each cell as the request its candidate
    answers, and the task those requests name, whole, because the machine that runs them has
    no database to look it up in. Complete enough to be held to, too: the revision of the code
    the cells are to be run with, so a machine on other code stops before it spends anything.

    One order is one kind of process. The cells that need the training stack and the ones that
    must run without it are never in one order, because on some platforms the two cannot share
    a process at all.

    Invariants: at least one cell, none twice; every request is over the order's task; every
    cell needs the same stack; the commit is non-empty without surrounding whitespace.

    Attributes:
        campaign: The campaign the cells belong to.
        task: The task every cell is learnt and scored on.
        evaluations: What each cell asks of its candidate, in the grid's order.
        git_commit: Revision of the code the cells are to be run with.
    """

    campaign: CampaignId
    task: DownstreamTask
    evaluations: tuple[CandidateEvaluation, ...]
    git_commit: str

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise InvalidCampaignOrderError(f"an order of campaign {self.campaign} names no cell")
        cells = [evaluation.cell for evaluation in self.evaluations]
        if len(set(cells)) != len(cells):
            raise InvalidCampaignOrderError("an order names a cell twice")
        if any(evaluation.task != self.task.task_id for evaluation in self.evaluations):
            raise InvalidCampaignOrderError(f"every cell of an order is over {self.task.task_id}")
        if len({e.declared.kind.needs_the_ml_stack for e in self.evaluations}) > 1:
            raise InvalidCampaignOrderError(
                "an order holds cells that need the training stack beside cells that must run "
                "without it"
            )
        if not self.git_commit or self.git_commit != self.git_commit.strip():
            raise InvalidCampaignOrderError(
                "git_commit must be non-empty without surrounding whitespace"
            )

    @property
    def needs_the_ml_stack(self) -> bool:
        """Whether the process that runs this order has to carry the training stack."""
        return self.evaluations[0].declared.kind.needs_the_ml_stack

    @property
    def cells(self) -> tuple[CampaignCell, ...]:
        return tuple(evaluation.cell for evaluation in self.evaluations)

    def require_commit(self, git_commit: str) -> None:
        """Stop a run on code other than the code ordered, before it runs a cell.

        Raises:
            CampaignOrderRejectedError: If ``git_commit`` is not the revision the order names.
        """
        if git_commit != self.git_commit:
            raise CampaignOrderRejectedError(
                f"this machine runs commit {git_commit}, the order is for commit {self.git_commit}"
            )

    def require_answered_by(self, result: CampaignOrderResult) -> None:
        """Refuse a result that answers another campaign or a cell this order never held.

        Raises:
            CampaignOrderRejectedError: If the result is of another campaign or commit, or
                answers a cell outside the order.
        """
        if result.campaign != self.campaign or result.git_commit != self.git_commit:
            raise CampaignOrderRejectedError(
                f"a result of campaign {result.campaign} at {result.git_commit} does not answer "
                f"an order of campaign {self.campaign} at {self.git_commit}"
            )
        foreign = {r.cell for r in result.results} - set(self.cells)
        if foreign:
            raise CampaignOrderRejectedError(
                f"the result answers {len(foreign)} cells the order never held"
            )
