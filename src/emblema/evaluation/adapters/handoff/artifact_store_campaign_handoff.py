import logging

from emblema.evaluation.adapters.documents.campaign_order_json import CampaignOrderJson
from emblema.evaluation.adapters.documents.campaign_order_result_json import (
    CampaignOrderResultJson,
)
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.exceptions import ArtifactNotFoundError

logger = logging.getLogger(__name__)


class ArtifactStoreCampaignHandoff:
    """Orders of cells and their results as JSON documents in the artifact store.

    The store the corpora and the fitted candidates travel through, so the machine that runs an
    order needs one set of credentials and one reference in each direction. Every report is
    logged with its reference as it is stored: on a platform that ends sessions without warning,
    the log is where the last result to resume from, or to accept, is read. Progress is stored
    transient and the closing result durable, so the many partial results of a long lane expire
    with the store's lifecycle rule instead of piling up.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._orders = CampaignOrderJson()
        self._results = CampaignOrderResultJson()

    def place(self, order: CampaignOrder) -> ArtifactRef:
        return self._store.put(self._orders.encode(order), Retention.DURABLE)

    def read_order(self, ref: ArtifactRef) -> CampaignOrder:
        return self._orders.decode(self._store.get(ref))

    def report_progress(self, result: CampaignOrderResult) -> ArtifactRef:
        return self._reported(result, Retention.TRANSIENT)

    def report(self, result: CampaignOrderResult) -> ArtifactRef:
        return self._reported(result, Retention.DURABLE)

    def read_progress(self, ref: ArtifactRef) -> CampaignOrderResult | None:
        try:
            return self.read_result(ref)
        except ArtifactNotFoundError:
            return None

    def read_result(self, ref: ArtifactRef) -> CampaignOrderResult:
        return self._results.decode(self._store.get(ref))

    def _reported(self, result: CampaignOrderResult, retention: Retention) -> ArtifactRef:
        ref = self._store.put(self._results.encode(result), retention)
        logger.info(
            "%d cells of campaign %s answered: %s %s",
            len(result.results),
            result.campaign,
            ref.key,
            ref.checksum,
        )
        return ref
