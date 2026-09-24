from dataclasses import dataclass

from emblema.evaluation.application.use_cases.record_cell_result import (
    RecordCellResult,
    RecordCellResultCommand,
)
from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class AcceptCampaignOrderResultCommand:
    """A result of an order, as the reference its machine reported it under.

    Attributes:
        result: The result to record.
    """

    result: ArtifactRef


class AcceptCampaignOrderResult:
    """Records what a machine outside this system answered of an order, cell by cell.

    The result is held to the order it names before anything is written: a result of another
    campaign, of other code, or answering a cell the order never held is refused whole. Each
    cell then goes the way a cell run by a worker goes, so a result accepted twice, or a cell a
    worker answered meanwhile, keeps the answer that stood first.
    """

    def __init__(self, handoff: CampaignHandoff, record_cell_result: RecordCellResult) -> None:
        self._handoff = handoff
        self._record = record_cell_result

    def __call__(self, command: AcceptCampaignOrderResultCommand) -> int:
        """Record every answered cell and return how many the result held.

        Raises:
            UnreadableCampaignDocumentError: If a reference holds no result or no order.
            CampaignOrderRejectedError: If the result does not answer the order it names.
            CampaignNotFoundError: If the campaign is unknown here.
            CampaignChangedElsewhereError: If the campaign kept moving under a write.
        """
        result = self._handoff.read_result(command.result)
        self._handoff.read_order(result.order).require_answered_by(result)
        for answered in result.results:
            self._record(RecordCellResultCommand(campaign=result.campaign, result=answered))
        return len(result.results)
