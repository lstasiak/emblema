from pathlib import Path
from typing import Protocol

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention


class ArtifactStore(Protocol):
    """Content-addressed store of binary artifacts.

    The key derives from the checksum of the content received, so a reference can never point at
    other bytes and the same content stored twice yields one reference; reads verify the bytes
    against it. Nothing is deleted: durable artifacts are the provenance of published results, and
    transient ones expire by lifecycle rule. Small artifacts pass as bytes; large ones, such as a
    tokenised corpus, as files through ``put_file`` and ``get_file``, never through a buffer of
    their size, which is also what a reader mapping an artifact into memory needs.
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
