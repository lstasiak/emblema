from typing import Protocol

from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.shared.kernel.artifacts import ArtifactRef


class CampaignHandoff(Protocol):
    """Where orders of cells and their results pass between this system and a machine outside it.

    An order goes out and a result comes back, each as a reference a person hands to the other
    side: the machine that runs the cells shares no database and no queue with this one, only
    what is stored. What is placed is read back unchanged, and a reference names one document
    for good, so a result can name the order it answers and be held to it.
    """

    def place(self, order: CampaignOrder) -> ArtifactRef:
        """Store the order and return the reference the other machine is handed."""
        ...

    def read_order(self, ref: ArtifactRef) -> CampaignOrder:
        """The order stored under ``ref``.

        Raises:
            UnreadableCampaignDocumentError: If what is stored there is not an order.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If what is stored does not hash to the reference's checksum.
        """
        ...

    def report(self, result: CampaignOrderResult) -> ArtifactRef:
        """Store what has been answered so far and return the reference that names it.

        Reporting the same result again stores nothing and returns the same reference.
        """
        ...

    def read_result(self, ref: ArtifactRef) -> CampaignOrderResult:
        """The result stored under ``ref``.

        Raises:
            UnreadableCampaignDocumentError: If what is stored there is not a result of an order.
            ArtifactNotFoundError: If nothing is stored under the reference.
            ArtifactIntegrityError: If what is stored does not hash to the reference's checksum.
        """
        ...
