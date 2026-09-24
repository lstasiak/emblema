from typing import Protocol

from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.shared.kernel.checksums import Checksum


class PromotableArtifactRepository(Protocol):
    """Keeps what finished campaigns kept: the projection a promotion is checked against.

    Stored under its origin, so the same announcement delivered twice leaves one artifact
    rather than two; that is what makes it safe to announce a campaign again after a delivery
    that failed.
    """

    def save(self, artifact: PromotableArtifact) -> None:
        """Store the artifact, replacing whatever an earlier announcement left under its origin."""
        ...

    def find_by_checksum(self, checksum: Checksum) -> tuple[PromotableArtifact, ...]:
        """Every artifact with that checksum, whichever campaign kept it; none if no campaign did.

        Ordered by when the campaign finished, then by campaign and competitor, so that the
        answer does not depend on the order announcements arrived in.
        """
        ...
