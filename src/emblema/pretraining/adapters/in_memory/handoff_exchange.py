from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError


class InMemoryHandoffExchange:
    """Exchange over two dictionaries: the fake of the port for application tests.

    Documents are kept as the values they are, under references minted from their text, so that
    the same order placed twice yields one reference as it would in a content-addressed store.
    """

    def __init__(self) -> None:
        self._documents: dict[ArtifactRef, PretrainingOrder | PretrainingResult] = {}

    def place(self, order: PretrainingOrder) -> ArtifactRef:
        return self._kept("order", order)

    def read_order(self, ref: ArtifactRef) -> PretrainingOrder:
        stored = self._found(ref)
        if not isinstance(stored, PretrainingOrder):
            raise UnreadableHandoffDocumentError(f"{ref.key!r} holds a result, not an order")
        return stored

    def report(self, result: PretrainingResult) -> ArtifactRef:
        return self._kept("result", result)

    def read_result(self, ref: ArtifactRef) -> PretrainingResult:
        stored = self._found(ref)
        if not isinstance(stored, PretrainingResult):
            raise UnreadableHandoffDocumentError(f"{ref.key!r} holds an order, not a result")
        return stored

    def _kept(self, kind: str, document: PretrainingOrder | PretrainingResult) -> ArtifactRef:
        checksum = Checksum.of_bytes(repr(document).encode())
        ref = ArtifactRef(f"durable/{kind}/{checksum.digest}", checksum)
        self._documents.setdefault(ref, document)
        return ref

    def _found(self, ref: ArtifactRef) -> PretrainingOrder | PretrainingResult:
        try:
            return self._documents[ref]
        except KeyError:
            raise ArtifactNotFoundError(ref.key) from None
