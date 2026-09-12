"""The artifact store port and the retention vocabulary its callers choose from."""

from enum import StrEnum
from pathlib import Path
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

    Artifacts come in two sizes and the port carries both. Small ones — a manifest, a
    configuration — pass as bytes. Ones an author would not want twice in memory, such as a
    tokenised corpus, pass as a file: ``put_file`` and ``get_file`` move them between the store
    and the local filesystem, never through a buffer the size of the artifact. A file is also
    what a reader that maps an artifact into memory needs, which no stream can stand in for.
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

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        """Store the contents of ``source`` and return the reference that resolves to them.

        Equivalent to ``put`` over the same bytes, down to the reference, but the file is read
        a chunk at a time. The file is left where it is.

        Raises:
            FileNotFoundError: If ``source`` does not exist.
        """
        ...

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        """Write the artifact ``ref`` points to at ``destination``, replacing what is there.

        The artifact appears at the destination whole or not at all, and only once its checksum
        has been verified, so a reader may map the file without checking it again.

        Raises:
            ArtifactNotFoundError: If nothing is stored under the reference's key.
            ArtifactIntegrityError: If the stored bytes do not hash to the reference's checksum.
        """
        ...

    def exists(self, ref: ArtifactRef) -> bool: ...
