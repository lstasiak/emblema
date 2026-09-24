from dataclasses import dataclass

from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.exceptions import CampaignOrderRejectedError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.evaluation.ports.candidate_provider import CandidateProvider
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class FulfilCampaignOrderCommand:
    """An order to run here, the code this machine runs, and what an earlier session finished.

    Attributes:
        order: The order to run.
        git_commit: Revision of the code on this machine.
        resume: The last result an interrupted session of the same order reported; its cells
            are kept and not run again. ``None`` to start from the first cell.
    """

    order: ArtifactRef
    git_commit: str
    resume: ArtifactRef | None = None


class FulfilCampaignOrder:
    """Runs an order's cells on a machine that has no registry, reporting after every one.

    The cells are answered by the same candidates a worker holds; what differs is that the task
    arrives with the order and the results leave through the store. The task is put where the
    candidates look tasks up, which on such a machine is a registry of this one run.

    Every cell answered is reported at once, as a result holding all cells so far, because the
    platforms this runs on end sessions without warning: whatever was finished is then behind
    the last reference reported, and a session started again from it runs only the rest.
    """

    def __init__(
        self,
        handoff: CampaignHandoff,
        tasks: DownstreamTaskRepository,
        candidates: CandidateProvider,
    ) -> None:
        self._handoff = handoff
        self._tasks = tasks
        self._candidates = candidates

    def __call__(self, command: FulfilCampaignOrderCommand) -> ArtifactRef:
        """Run every cell not already answered and return the reference of the whole result.

        Raises:
            CampaignOrderRejectedError: If this machine runs other code than the order names,
                or the result resumed from answers another order.
            UnreadableCampaignDocumentError: If a reference holds no order or no result.
            UnknownCandidateError: If this process supplies no candidate a cell names.
            CandidateMismatchError: If a candidate here is not what the campaign recorded.
        """
        order = self._handoff.read_order(command.order)
        order.require_commit(command.git_commit)
        self._tasks.save(order.task)
        answered = (
            []
            if command.resume is None
            else list(self._resumed(command.order, command.resume).results)
        )
        for evaluation in order.evaluations:
            if any(result.cell == evaluation.cell for result in answered):
                continue
            answered.append(self._candidates.evaluate(evaluation))
            self._handoff.report(self._result(command, order, answered))
        # Everything was reported cell by cell already; this names the whole, and stores
        # nothing new unless the order was resumed with every cell answered.
        return self._handoff.report(self._result(command, order, answered))

    @staticmethod
    def _result(
        command: FulfilCampaignOrderCommand, order: CampaignOrder, answered: list[CellResult]
    ) -> CampaignOrderResult:
        return CampaignOrderResult(
            order=command.order,
            campaign=order.campaign,
            git_commit=order.git_commit,
            results=tuple(answered),
        )

    def _resumed(self, order: ArtifactRef, resume: ArtifactRef) -> CampaignOrderResult:
        """The result an interrupted session left, held to the order it claims to answer.

        Raises:
            CampaignOrderRejectedError: If it answers another order.
        """
        earlier = self._handoff.read_result(resume)
        if earlier.order != order:
            raise CampaignOrderRejectedError(
                f"the result resumed from answers order {earlier.order.key}, not {order.key}"
            )
        return earlier
