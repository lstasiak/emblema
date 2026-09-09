"""The artifact store port and the retention vocabulary its callers choose from."""

from enum import StrEnum
from typing import Protocol

from emblema.shared.kernel.artifacts import ArtifactRef


class Retention(StrEnum):
    """How long a stored artifact is kept.

    Attributes:
        DURABLE: Kept until removed on purpose. The class of every artifact the domain
            registers, so a reference recorded in the database keeps resolving.
        TRANSIENT: Intermediate output such as a checkpoint written mid-training. The store
            may expire it by a lifecycle rule once it is older than the configured age.
    """

    DURABLE = "durable"
    TRANSIENT = "transient"


class ArtifactStore(Protocol):
    """Content-addressed store of binary artifacts.

    The store computes the checksum of the content it receives and derives the key from it,
    so an existing reference can never come to point at different bytes and the same content
    stored twice yields the same reference. Reading verifies the bytes against the reference's
    checksum before returning them. There is no way to delete: durable artifacts are the
    provenance of every published result, and transient ones expire by lifecycle rule.
    """

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        """Store ``content`` and return the reference that resolves to it.

        Storing content that is already present writes nothing and returns the existing
        reference.
        """
        ...

    def get(self, ref: ArtifactRef) -> bytes:
        """Return the bytes ``ref`` points to.

        Raises:
            ArtifactNotFoundError: If nothing is stored under the reference's key.
            ArtifactIntegrityError: If the stored bytes do not hash to the reference's checksum.
        """
        ...

    def exists(self, ref: ArtifactRef) -> bool: ...
