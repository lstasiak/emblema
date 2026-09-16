from typing import Protocol

from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.shared.kernel.artifacts import ArtifactRef


class HandoffExchange(Protocol):
    """Where orders and results pass between this system and the machine that trains for it.

    An order goes out and a result comes back, each as a reference the other side is handed by
    a person: the two machines share no process, no database and no clock, only what is stored.
    What is placed is read back unchanged, and a reference names one document for good, so a
    result can point at the order it fulfilled and be held to it later.
    """

    def place(self, order: PretrainingOrder) -> ArtifactRef:
        """Store the order and return the reference the other machine is handed."""
        ...

    def read_order(self, ref: ArtifactRef) -> PretrainingOrder:
        """The order stored under ``ref``.

        Raises:
            UnreadableHandoffDocumentError: If what is stored there is not an order.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If what is stored does not hash to the reference's checksum.
        """
        ...

    def report(self, result: PretrainingResult) -> ArtifactRef:
        """Store the result and return the reference this system is handed back."""
        ...

    def read_result(self, ref: ArtifactRef) -> PretrainingResult:
        """The result stored under ``ref``.

        Raises:
            UnreadableHandoffDocumentError: If what is stored there is not a result.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If what is stored does not hash to the reference's checksum.
        """
        ...
