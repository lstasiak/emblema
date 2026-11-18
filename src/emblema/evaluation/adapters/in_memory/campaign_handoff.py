from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError


class InMemoryCampaignHandoff:
    """Handoff over one dictionary: the fake of the port for application tests.

    Documents are kept as the values they are, under references minted from their text, so that
    the same result reported twice yields one reference as it would in a content-addressed store.
    Progress and the closing result sit under different keys, as they do in the store.
    """

    def __init__(self) -> None:
        self._documents: dict[ArtifactRef, CampaignOrder | CampaignOrderResult] = {}
        self.reported: list[ArtifactRef] = []

    def place(self, order: CampaignOrder) -> ArtifactRef:
        return self._kept(Retention.DURABLE, "order", order)

    def read_order(self, ref: ArtifactRef) -> CampaignOrder:
        stored = self._found(ref)
        if not isinstance(stored, CampaignOrder):
            raise UnreadableCampaignDocumentError(f"{ref.key!r} holds a result, not an order")
        return stored

    def report_progress(self, result: CampaignOrderResult) -> ArtifactRef:
        ref = self._kept(Retention.TRANSIENT, "result", result)
        self.reported.append(ref)
        return ref

    def report(self, result: CampaignOrderResult) -> ArtifactRef:
        ref = self._kept(Retention.DURABLE, "result", result)
        self.reported.append(ref)
        return ref

    def read_progress(self, ref: ArtifactRef) -> CampaignOrderResult | None:
        try:
            return self.read_result(ref)
        except ArtifactNotFoundError:
            return None

    def read_result(self, ref: ArtifactRef) -> CampaignOrderResult:
        stored = self._found(ref)
        if not isinstance(stored, CampaignOrderResult):
            raise UnreadableCampaignDocumentError(f"{ref.key!r} holds an order, not a result")
        return stored

    def _kept(
        self, retention: Retention, kind: str, document: CampaignOrder | CampaignOrderResult
    ) -> ArtifactRef:
        checksum = Checksum.of_bytes(repr(document).encode())
        ref = ArtifactRef(f"{retention}/campaign-{kind}/{checksum.digest}", checksum)
        self._documents.setdefault(ref, document)
        return ref

    def _found(self, ref: ArtifactRef) -> CampaignOrder | CampaignOrderResult:
        try:
            return self._documents[ref]
        except KeyError:
            raise ArtifactNotFoundError(ref.key) from None
