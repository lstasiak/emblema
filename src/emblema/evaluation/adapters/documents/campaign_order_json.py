import json
from typing import Any, Final

from emblema.evaluation.adapters.documents.campaign_candidate_document import (
    CampaignCandidateDocument,
)
from emblema.evaluation.adapters.documents.cell_result_document import CellResultDocument
from emblema.evaluation.adapters.documents.downstream_task_document import DownstreamTaskDocument
from emblema.evaluation.adapters.documents.stored_document import MALFORMED, read_document
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose


class CampaignOrderJson:
    """An order of campaign cells as JSON, the form it is stored in and handed to another machine.

    Compact and key-ordered, so the same order is always the same bytes and placing it twice
    stores one document.
    """

    FORMAT: Final = "emblema.campaign-order"
    # Bumped when a document of the version before can no longer be read here.
    VERSION: Final = 1

    def __init__(self) -> None:
        self._tasks = DownstreamTaskDocument()
        self._candidates = CampaignCandidateDocument()

    def encode(self, order: CampaignOrder) -> bytes:
        document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "campaign": str(order.campaign),
            "task": self._tasks.encode(order.task),
            "evaluations": [self._evaluation(evaluation) for evaluation in order.evaluations],
            "git_commit": order.git_commit,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> CampaignOrder:
        """The order those bytes state.

        Raises:
            UnreadableCampaignDocumentError: If the bytes are not an order of a version this
                reads, a field is missing or mistyped, or what is stated is not an order.
        """
        document = read_document(content, self.FORMAT, self.VERSION, "order")
        try:
            return CampaignOrder(
                campaign=CampaignId.parse(document["campaign"]),
                task=self._tasks.decode(document["task"]),
                evaluations=tuple(self._read(e) for e in document["evaluations"]),
                git_commit=document["git_commit"],
            )
        except MALFORMED as error:
            raise UnreadableCampaignDocumentError(f"order is not well formed: {error}") from error

    def _evaluation(self, evaluation: CandidateEvaluation) -> dict[str, Any]:
        return {
            "task": str(evaluation.task),
            "cell": CellResultDocument.encode_cell(evaluation.cell),
            "purpose": evaluation.purpose.value,
            "retain": evaluation.retain,
            "declared": self._candidates.encode(evaluation.declared),
            "holdout": None if evaluation.holdout is None else evaluation.holdout.one_in,
        }

    def _read(self, document: dict[str, Any]) -> CandidateEvaluation:
        return CandidateEvaluation(
            task=TaskId.parse(document["task"]),
            cell=CellResultDocument.decode_cell(document["cell"]),
            purpose=RunPurpose(document["purpose"]),
            retain=bool(document["retain"]),
            declared=self._candidates.decode(document["declared"]),
            holdout=None
            if document["holdout"] is None
            else InnerHoldout(one_in=document["holdout"]),
        )
