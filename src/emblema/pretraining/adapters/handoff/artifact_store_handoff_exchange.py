from emblema.pretraining.adapters.documents.pretraining_order_json import PretrainingOrderJson
from emblema.pretraining.adapters.documents.pretraining_result_json import PretrainingResultJson
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention


class ArtifactStoreHandoffExchange:
    """Orders and results as JSON documents in the artifact store.

    The same store the weights and the corpus travel through, so the machine that trains needs
    one set of credentials and one reference in each direction; content-addressed, so a result
    names the order it fulfilled by a reference nothing can quietly change under. Both documents
    are durable: they are the provenance of a backbone, not an intermediate of the run.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._orders = PretrainingOrderJson()
        self._results = PretrainingResultJson()

    def place(self, order: PretrainingOrder) -> ArtifactRef:
        return self._store.put(self._orders.encode(order), Retention.DURABLE)

    def read_order(self, ref: ArtifactRef) -> PretrainingOrder:
        return self._orders.decode(self._store.get(ref))

    def report(self, result: PretrainingResult) -> ArtifactRef:
        return self._store.put(self._results.encode(result), Retention.DURABLE)

    def read_result(self, ref: ArtifactRef) -> PretrainingResult:
        return self._results.decode(self._store.get(ref))
