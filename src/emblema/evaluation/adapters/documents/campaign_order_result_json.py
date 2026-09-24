import json
from typing import Final

from emblema.evaluation.adapters.documents.cell_result_document import CellResultDocument
from emblema.evaluation.adapters.documents.stored_document import MALFORMED, read_document
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CampaignOrderResultJson:
    """What a machine answered of an order, as JSON, the form it comes back through the store in.

    Compact and key-ordered, so reporting the same cells twice stores one document.
    """

    FORMAT: Final = "emblema.campaign-order-result"
    # Bumped when a document of the version before can no longer be read here.
    VERSION: Final = 1

    def __init__(self) -> None:
        self._results = CellResultDocument()

    def encode(self, result: CampaignOrderResult) -> bytes:
        document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "order": {
                "key": result.order.key,
                "algorithm": str(result.order.checksum.algorithm),
                "digest": result.order.checksum.digest,
            },
            "campaign": str(result.campaign),
            "git_commit": result.git_commit,
            "results": [self._results.encode(answered) for answered in result.results],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> CampaignOrderResult:
        """The result those bytes state.

        Raises:
            UnreadableCampaignDocumentError: If the bytes are not a result of a version this
                reads, a field is missing or mistyped, or what is stated is not a result.
        """
        document = read_document(content, self.FORMAT, self.VERSION, "result of an order")
        try:
            order = document["order"]
            return CampaignOrderResult(
                order=ArtifactRef(
                    order["key"], Checksum(HashAlgorithm(order["algorithm"]), order["digest"])
                ),
                campaign=CampaignId.parse(document["campaign"]),
                git_commit=document["git_commit"],
                results=tuple(self._results.decode(answered) for answered in document["results"]),
            )
        except MALFORMED as error:
            raise UnreadableCampaignDocumentError(
                f"result of an order is not well formed: {error}"
            ) from error
