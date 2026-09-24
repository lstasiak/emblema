from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError


class InMemoryCampaignHandoff:
    """Handoff over one dictionary: the fake of the port for application tests.

    Documents are kept as the values they are, under references minted from their text, so that
    the same result reported twice yields one reference as it would in a content-addressed store.
    """

    def __init__(self) -> None:
        self._documents: dict[ArtifactRef, CampaignOrder | CampaignOrderResult] = {}
        self.reported: list[ArtifactRef] = []

    def place(self, order: CampaignOrder) -> ArtifactRef:
        return self._kept("order", order)

    def read_order(self, ref: ArtifactRef) -> CampaignOrder:
        stored = self._found(ref)
        if not isinstance(stored, CampaignOrder):
            raise UnreadableCampaignDocumentError(f"{ref.key!r} holds a result, not an order")
        return stored

    def report(self, result: CampaignOrderResult) -> ArtifactRef:
        ref = self._kept("result", result)
        self.reported.append(ref)
        return ref

    def read_result(self, ref: ArtifactRef) -> CampaignOrderResult:
        stored = self._found(ref)
        if not isinstance(stored, CampaignOrderResult):
            raise UnreadableCampaignDocumentError(f"{ref.key!r} holds an order, not a result")
        return stored

    def _kept(self, kind: str, document: CampaignOrder | CampaignOrderResult) -> ArtifactRef:
        checksum = Checksum.of_bytes(repr(document).encode())
        ref = ArtifactRef(f"durable/campaign-{kind}/{checksum.digest}", checksum)
        self._documents.setdefault(ref, document)
        return ref

    def _found(self, ref: ArtifactRef) -> CampaignOrder | CampaignOrderResult:
        try:
            return self._documents[ref]
        except KeyError:
            raise ArtifactNotFoundError(ref.key) from None
